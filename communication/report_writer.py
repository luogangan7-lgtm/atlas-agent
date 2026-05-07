"""Daily and weekly/evolution report writer."""
import logging
from datetime import datetime, timezone
from pathlib import Path

from storage import state_manager as sm

LOG = logging.getLogger("atlas.report")

VAULT      = Path("/Volumes/data/obsidian-vault")
BRAIN_DIR  = VAULT / "agent-brain"
DAILY_DIR  = BRAIN_DIR / "daily"
WEEKLY_DIR = BRAIN_DIR / "weekly"


def write_daily_report(cycle_results: list[dict], world_state_summary: dict) -> str | None:
    now  = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    ctx    = sm.load_context()
    model  = sm.load_self_model()

    done_lines = []
    for r in cycle_results:
        t = r.get("type", "")
        if t == "l0_organize":
            done_lines.append(f"- 处理了 {r.get('processed', 0)} 条 L0 原料，生成 {r.get('nodes_created', 0)} 个 L1 知识节点")
        elif t == "l1_complete":
            done_lines.append(f"- 补全了 {r.get('completed', 0)} 条 L1 知识（不完整→完整）")
        elif t == "l2_generate":
            done_lines.append(f"- 生成了 {r.get('generated', 0)} 条 L2 跨域洞见")
        elif t == "l3_synthesize":
            done_lines.append(f"- 蒸馏了 {r.get('synthesized', 0)} 个 L3 框架")
        elif t == "contradiction_detect":
            n = r.get("contradictions", 0)
            if n: done_lines.append(f"- 发现 {n} 处知识矛盾，已写入收件箱")
        elif t == "autonomous_research":
            n = r.get("nodes_queued", 0)
            d = r.get("domain", "")
            if n: done_lines.append(f"- 自主研究了「{d}」领域，生成 {n} 个待审核知识节点")
        elif t == "l4_feedback":
            n = r.get("l1_enriched", 0)
            if n: done_lines.append(f"- L4 元规律反哺了 {n} 条 L1 知识（跨层洞见）")

    if not done_lines:
        done_lines = ["- 本周期无重大操作（知识库状态良好）"]

    kh  = world_state_summary.get("knowledge_health", {})
    l0  = kh.get("pending_l0", 0)
    l1  = kh.get("l1_total", 0)
    l1i = kh.get("incomplete_l1", 0)
    l2  = kh.get("l2_total", 0)
    l3  = kh.get("l3_total", 0)
    l4  = kh.get("l4_total", 0)

    gaps     = world_state_summary.get("knowledge_gaps", [])
    gaps_str = "、".join(gaps[:5]) if gaps else "无"
    pending  = ctx.get("pending_user_decisions", 0)

    # Self-model summary
    prefs = model.get("behavioral_preferences", {})
    strong = "、".join(prefs.get("preferred_domains", [])[:3]) or "积累中"
    weak   = "、".join(prefs.get("weak_domains", [])[:3]) or "暂无"

    # Health from context
    health_score = world_state_summary.get("health_score", "")
    health_grade = world_state_summary.get("health_grade", "")
    health_line  = f"健康评分：{health_grade}（{health_score:.0%}）" if health_score else ""

    content = f"""---
date: {date}
type: daily_report
agent: atlas-knowledge-agent
---

# 代理日报 - {date}

## 今天我做了什么

{"".join(line + "\n" for line in done_lines)}
## 知识库健康状况

| 层级 | 数量 |
|------|------|
| L0 待处理 | {l0} |
| L1 知识节点 | {l1}（其中 {l1i} 条待补全） |
| L2 跨域洞见 | {l2} |
| L3 框架 | {l3} |
| L4 元规律 | {l4} |

{health_line}
知识薄弱域（<5条L1）：{gaps_str}

## 我对自己的认知

- 强项领域：{strong}
- 待强化：{weak}

## 待你决策

{"有 **" + str(pending) + "** 条事项需要你决策，请查看 `agent-brain/inbox/`" if pending > 0 else "暂无需要你决策的事项。"}

## 明天我打算做什么

{"- 继续补全待完善 L1（" + str(l1i) + " 条）" if l1i > 0 else ""}
{"- 处理新 L0 原料（" + str(l0) + " 条待处理）" if l0 > 0 else ""}
{"- 补充薄弱域：" + gaps_str if gaps else ""}
- 持续生成 L2 跨域洞见
- 在进化周期运行 L4→L1 反哺和 Prompt 优化

---
*由 ATLAS 知识代理自动生成 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""

    report_path = DAILY_DIR / f"{date}.md"
    report_path.write_text(content, encoding="utf-8")
    LOG.info(f"Daily report written: agent-brain/daily/{date}.md")
    return f"agent-brain/daily/{date}.md"


def write_evolution_report(weekly_stats: dict, health, self_model: dict,
                           prompt_results: dict, strategy_results: dict,
                           l4_results: dict) -> str | None:
    now  = datetime.now(timezone.utc)
    week = now.strftime("%Y-W%V")
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    prefs   = self_model.get("behavioral_preferences", {})
    strong  = "、".join(prefs.get("preferred_domains", [])[:5]) or "积累中"
    weak    = "、".join(prefs.get("weak_domains", [])[:5]) or "暂无"
    curious = "、".join(prefs.get("curiosity_domains", [])[:3]) or "暂无"

    cap  = self_model.get("capability_assessment", {})
    cap_lines = []
    for task, stats in cap.items():
        cap_lines.append(f"- {task}: 运行{stats['runs']}次，成功率{stats['success_rate']:.0%}，平均产出{stats['avg_output']:.1f}")

    # Health
    hg = getattr(health, "grade", "?")
    hs = getattr(health, "score", 0)
    hl1 = getattr(health, "l1_total", 0)
    hl2 = getattr(health, "l2_total", 0)
    hl3 = getattr(health, "l3_total", 0)
    hl4 = getattr(health, "l4_total", 0)
    hcov = getattr(health, "coverage_score", 0)
    hcomp = getattr(health, "avg_l1_completeness", 0)
    hgaps = getattr(health, "gap_domains", [])

    # Strategy adjustments
    adj_reasoning = []
    if isinstance(strategy_results, dict):
        adj_reasoning = strategy_results.get("reasoning", [])
    from storage import state_manager as sm
    strat = sm.load("strategy", {})
    adj_reasoning = strat.get("reasoning", [])
    adj_str = "\n".join(f"- {r}" for r in adj_reasoning) if adj_reasoning else "- 无需调整"

    # Prompt status
    prompts_file = Path("/Volumes/data/obsidian-vault/agent-brain/state/prompts.json")
    prompt_str = "- 无变化"
    if prompts_file.exists():
        import json
        try:
            prompts = json.loads(prompts_file.read_text("utf-8"))
            prompt_str = "\n".join(
                f"- {name}: {p.get('version','?')} (质量分 {p.get('quality_score',0):.0%})"
                for name, p in prompts.items()
            )
        except Exception:
            pass

    content = f"""---
week: {week}
type: evolution_report
agent: atlas-knowledge-agent
health_grade: {hg}
health_score: {hs:.3f}
---

# 进化周报 - {week}

## 本周知识成长

| 指标 | 数值 |
|------|------|
| 新增知识节点 | {weekly_stats.get("nodes_created", 0)} |
| L1 补全 | {weekly_stats.get("l1_completed", 0)} |
| L2 跨域洞见 | {weekly_stats.get("l2_generated", 0)} |
| L3 框架蒸馏 | {weekly_stats.get("l3_synthesized", 0)} |
| L4 反哺 L1 | {l4_results.get("l1_enriched", 0)} |

## 知识库健康评估

**整体评分：{hg}（{hs:.0%}）**

| 层级 | 数量 |
|------|------|
| L1 | {hl1} |
| L2 | {hl2} |
| L3 | {hl3} |
| L4 | {hl4} |

- L1→L2 覆盖率：{hcov:.1%}（目标 >30%）
- L1 平均完整度：{hcomp:.1%}
- 薄弱域（前5）：{"、".join(hgaps[:5]) or "无"}

## 我的能力评估

{chr(10).join(cap_lines) or "- 数据积累中"}

## 我对各领域的理解

- **强项**：{strong}
- **好奇**：{curious}
- **弱项（待补）**：{weak}

## Prompt 进化记录

{prompt_str}

## 策略调整

{adj_str}

## 下周方向

- 重点强化薄弱域知识
- 持续 L4→L1 反哺，提升跨层理解
- 探索新的跨域连接机会

---
*由 ATLAS 知识代理自动生成 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""

    report_path = WEEKLY_DIR / f"{week}.md"
    report_path.write_text(content, encoding="utf-8")
    LOG.info(f"Evolution report written: agent-brain/weekly/{week}.md")
    return f"agent-brain/weekly/{week}.md"


# Keep old weekly report function for compatibility
def write_weekly_report(stats: dict) -> str | None:
    return write_evolution_report(stats, None, sm.load_self_model(), {}, {}, {})
