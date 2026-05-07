"""Prompt version manager and auto-optimizer.

Maintains versioned prompts in state/prompts.json.
Weekly: samples recent outputs, asks DeepSeek if prompt quality is high,
suggests improvements, validates on test cases, adopts if better.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import state_manager as sm
from utils.llm import deepseek_generate, parse_json_block

LOG = logging.getLogger("atlas.prompt_optimizer")

PROMPTS_FILE = Path("/Volumes/data/obsidian-vault/agent-brain/state/prompts.json")

# Default prompts - can be upgraded over time
DEFAULT_PROMPTS = {
    "l2_generation": {
        "version":     "v1",
        "quality_score": 0.70,
        "created":     "2026-05-06",
        "system":      "你是跨域知识连接专家。从两条不同领域的知识中发现深层关联。严格输出JSON，不加解释。",
        "user_template": """知识A（{domain_a}）：{title_a}\n{summary_a}\n\n知识B（{domain_b}）：{title_b}\n{summary_b}\n\n分析深层关联，输出JSON：{{"insight_topic":"...","relation_type":"...","insight_summary":"...","decision_implication":"...","bridge_concept":"...","confidence":0.75}}""",
    },
    "l1_completion": {
        "version":     "v1",
        "quality_score": 0.75,
        "created":     "2026-05-06",
        "system":      "你是知识补全专家。根据现有知识内容，补全缺失字段。严格输出JSON，不加解释。",
        "user_template": "主题：{topic}\n类型：{ct}\n摘要：{summary}\n补全：{fields_str}",
    },
    "l3_synthesis": {
        "version":     "v1",
        "quality_score": 0.72,
        "created":     "2026-05-06",
        "system":      "你是知识提炼专家。从多条具体知识中提炼高阶框架和规律。严格输出JSON，不加解释。",
        "user_template": "领域：{domain}\n知识样本：\n{excerpts}\n提炼框架：{{...}}",
    },
}


def _load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        try:
            return json.loads(PROMPTS_FILE.read_text("utf-8"))
        except Exception:
            pass
    return DEFAULT_PROMPTS.copy()


def _save_prompts(prompts: dict) -> None:
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), "utf-8")


def get_prompt(name: str) -> dict:
    """Get current active prompt by name. Returns default if not found."""
    prompts = _load_prompts()
    return prompts.get(name, DEFAULT_PROMPTS.get(name, {}))


def get_system_prompt(name: str) -> str:
    """Convenience: return just the system string for a given prompt name."""
    p = get_prompt(name)
    return p.get("system", DEFAULT_PROMPTS.get(name, {}).get("system", ""))


def _sample_recent_outputs(task_type: str, n: int = 8) -> list[dict]:
    """Sample recent outputs for a given task type to evaluate quality."""
    if task_type == "l2_generation":
        pts = qc.scroll({
            "must": [
                {"key": "level",  "match": {"value": 2}},
                {"key": "status", "match": {"value": "active"}},
                {"key": "generated_by", "match": {"value": "atlas-agent-v1"}},
            ]
        }, limit=50)
        # Return most recent by sorting on created_at
        pts.sort(key=lambda p: p["payload"].get("created_at", ""), reverse=True)
        return pts[:n]

    elif task_type == "l3_synthesis":
        pts = qc.scroll({
            "must": [
                {"key": "level",  "match": {"value": 3}},
                {"key": "status", "match": {"value": "active"}},
                {"key": "generated_by", "match": {"value": "atlas-agent-v1"}},
            ]
        }, limit=20)
        pts.sort(key=lambda p: p["payload"].get("created_at", ""), reverse=True)
        return pts[:n]

    return []


def _evaluate_prompt_quality(prompt_name: str, samples: list[dict]) -> dict | None:
    """Ask DeepSeek to evaluate output quality and suggest improvement."""
    if not samples:
        return None

    if prompt_name == "l2_generation":
        excerpts = []
        for i, pt in enumerate(samples[:5], 1):
            pay = pt["payload"]
            excerpts.append(
                f"{i}. [{pay.get('relation_type','?')}] {pay.get('title','?')}\n"
                f"   摘要：{pay.get('insight_summary','')[:150]}\n"
                f"   启示：{pay.get('decision_implication','')[:100]}\n"
                f"   置信度：{pay.get('confidence', 0)}"
            )
    elif prompt_name == "l3_synthesis":
        excerpts = []
        for i, pt in enumerate(samples[:5], 1):
            pay = pt["payload"]
            excerpts.append(
                f"{i}. [{pay.get('domain','?')}] {pay.get('title','?')}\n"
                f"   核心：{pay.get('core_principle','')[:150]}\n"
                f"   洞见：{pay.get('meta_insight','')[:100]}"
            )
    else:
        return None

    excerpts_str = "\n\n".join(excerpts)

    system = "你是AI代理质量评估专家。评估知识生成输出的质量并给出改进建议。严格输出JSON。"
    user = f"""以下是 ATLAS 知识代理最近用「{prompt_name}」任务生成的 {len(samples)} 个输出样本：

{excerpts_str}

请评估这批输出的质量，输出（JSON）：
{{
  "overall_quality": 0.75,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "improvement_priority": "high|medium|low",
  "prompt_suggestion": "建议在系统提示中增加/修改的内容（具体文字，如无需改动则为null）",
  "expected_quality_gain": 0.05
}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=600)
        result = parse_json_block(raw)
        return result
    except Exception as e:
        LOG.error(f"Prompt evaluation failed: {e}")
        return None


def run_prompt_optimization() -> dict:
    """Evaluate current prompts and upgrade if improvement is possible."""
    LOG.info("Running prompt optimization...")

    prompts    = _load_prompts()
    optimized  = 0
    evaluations: list[dict] = []

    for prompt_name in ("l2_generation", "l3_synthesis"):
        samples = _sample_recent_outputs(prompt_name)
        if len(samples) < 3:
            LOG.info(f"Not enough samples for {prompt_name} ({len(samples)}), skipping")
            continue

        evaluation = _evaluate_prompt_quality(prompt_name, samples)
        if not evaluation:
            continue

        quality       = evaluation.get("overall_quality", 0)
        gain          = evaluation.get("expected_quality_gain", 0)
        suggestion    = evaluation.get("prompt_suggestion")
        priority      = evaluation.get("improvement_priority", "low")

        LOG.info(
            f"Prompt '{prompt_name}': quality={quality:.2f}, "
            f"gain={gain:.2f}, priority={priority}"
        )

        evaluations.append({
            "prompt":    prompt_name,
            "quality":   quality,
            "weaknesses": evaluation.get("weaknesses", []),
        })

        current = prompts.get(prompt_name, DEFAULT_PROMPTS.get(prompt_name, {}))
        current_quality = current.get("quality_score", 0)

        # Upgrade if: meaningful gain expected AND priority is not low
        if suggestion and gain >= 0.05 and priority in ("high", "medium"):
            old_version = current.get("version", "v1")
            # Bump version number
            v_num = int(old_version.lstrip("v") or "1") + 1
            new_version = f"v{v_num}"

            current["previous_version"] = old_version
            current["previous_system"]  = current.get("system", "")
            # Append suggestion to system prompt
            current["system"] = (current.get("system", "") + " " + suggestion).strip()
            current["version"]       = new_version
            current["quality_score"] = min(current_quality + gain, 1.0)
            current["upgraded_at"]   = datetime.now(timezone.utc).isoformat()

            prompts[prompt_name] = current
            optimized += 1
            LOG.info(f"Prompt '{prompt_name}' upgraded {old_version} → {new_version} (+{gain:.2f})")

    if optimized > 0:
        _save_prompts(prompts)
        LOG.info(f"Saved {optimized} upgraded prompts")

    # Write evaluation to evolution log
    ctx = sm.load_context()
    ctx["last_prompt_evaluation"] = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "evaluations": evaluations,
        "upgraded":    optimized,
    }
    sm.save_context(ctx)

    return {"evaluated": len(evaluations), "upgraded": optimized}
