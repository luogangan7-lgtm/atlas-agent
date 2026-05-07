"""Contradiction detector.

Algorithm:
1. Find L1 pairs with very high vector similarity (>0.88) but from different L0 sources
2. Use DeepSeek to check if their content contradicts each other
3. If contradiction found:
   - Confidence > 0.8: attempt auto-merge or flag one as superseded
   - Confidence ≤ 0.8: write to inbox/ for user decision
"""
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import state_manager as sm
from utils.llm import deepseek_generate, parse_json_block

LOG = logging.getLogger("atlas.contradiction")

VAULT       = Path("/Volumes/data/obsidian-vault")
INBOX_DIR   = VAULT / "agent-brain" / "inbox"
SIM_MIN     = 0.88   # high similarity → potentially same claim
MAX_CHECKS  = 5      # limit LLM calls per cycle


def _check_contradiction(node_a: dict, node_b: dict) -> dict | None:
    """Ask DeepSeek: are these two knowledge nodes contradictory?"""
    pay_a = node_a["payload"]
    pay_b = node_b["payload"]

    title_a  = pay_a.get("title") or pay_a.get("topic") or ""
    title_b  = pay_b.get("title") or pay_b.get("topic") or ""
    sum_a    = pay_a.get("summary") or ""
    sum_b    = pay_b.get("summary") or ""

    system = "你是知识一致性审查专家。分析两条知识是否相互矛盾。严格输出JSON，不加解释。"
    user = f"""知识A：{title_a}
{sum_a[:500]}

知识B：{title_b}
{sum_b[:500]}

请判断（严格JSON）：
{{
  "is_contradictory": true/false,
  "contradiction_type": "direct|conditional|scope_mismatch|none",
  "analysis": "分析说明（2-3句）",
  "resolution": "建议解决方案（合并/保留双方/选一/情境化）",
  "confidence": 0.8,
  "preferred_id": "A或B（如果一个明显更可信）或null"
}}

contradiction_type：
- direct: 直接矛盾，两者不能同时为真
- conditional: 条件性矛盾，适用场景不同各自成立
- scope_mismatch: 范围不同，表面矛盾实为不同层次
- none: 不矛盾（相似但互补）"""

    try:
        raw = deepseek_generate(system, user, max_tokens=500)
        result = parse_json_block(raw)
        return result if result else None
    except Exception as e:
        LOG.error(f"Contradiction check failed: {e}")
        return None


def _write_inbox_contradiction(node_a: dict, node_b: dict, analysis: dict) -> str:
    """Write a contradiction decision request to inbox/."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    now  = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d-%H%M%S")

    pay_a   = node_a["payload"]
    pay_b   = node_b["payload"]
    title_a = pay_a.get("title") or pay_a.get("topic") or str(node_a["id"])
    title_b = pay_b.get("title") or pay_b.get("topic") or str(node_b["id"])
    path_a  = pay_a.get("obsidian_path", "")
    path_b  = pay_b.get("obsidian_path", "")
    link_a  = f"[[{path_a}|{title_a}]]" if path_a else title_a
    link_b  = f"[[{path_b}|{title_b}]]" if path_b else title_b

    content = f"""---
type: user_decision_required
category: contradiction
priority: medium
created: {now.isoformat()}
status: pending
node_a_id: {node_a["id"]}
node_b_id: {node_b["id"]}
---

# 需要你决策：发现知识矛盾

## 矛盾类型
**{analysis.get("contradiction_type", "未知")}** — {analysis.get("analysis", "")}

## 知识 A（{pay_a.get("domain", "")}）
> {link_a}

{pay_a.get("summary", "")[:400]}

## 知识 B（{pay_b.get("domain", "")}）
> {link_b}

{pay_b.get("summary", "")[:400]}

## 我的分析
{analysis.get("analysis", "")}

## 我的建议
{analysis.get("resolution", "")}

---

## 请你决定（勾选后保存，下次循环我会处理）

- [ ] 接受建议：{analysis.get("resolution", "")}
- [ ] 保留 A，标记 B 为过时
- [ ] 保留 B，标记 A 为过时
- [ ] 两者都保留，各自标注适用场景
- [ ] 其他想法：

---
*由 ATLAS 知识代理自动生成 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""

    filename = f"contradiction-{date}.md"
    (INBOX_DIR / filename).write_text(content, encoding="utf-8")
    return f"agent-brain/inbox/{filename}"


def run_contradiction_detection(max_checks: int = MAX_CHECKS) -> dict:
    LOG.info(f"Contradiction detection: checking up to {max_checks} candidate pairs")

    # Sample L1 records with vectors
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=80, with_vector=True)

    pts = [p for p in pts if p.get("vector")]
    if len(pts) < 2:
        LOG.info("Not enough L1 records with vectors")
        return {"checked": 0, "contradictions": 0, "inbox_written": 0}

    random.shuffle(pts)
    seeds = pts[:20]  # check a random subset each cycle

    checked = 0
    contradictions = 0
    inbox_written = 0
    tried: set[tuple] = set()

    for seed in seeds:
        if checked >= max_checks:
            break

        candidates = qc.search(
            vector=seed["vector"],
            filter_body={
                "must": [
                    {"key": "level",  "match": {"value": 1}},
                    {"key": "status", "match": {"value": "active"}},
                ],
                "must_not": [
                    {"key": "record_type", "match": {"value": "entity"}},
                ],
            },
            limit=5,
            score_threshold=SIM_MIN,
        )
        # Exclude self
        candidates = [c for c in candidates if c["id"] != seed["id"]]

        for cand in candidates:
            if checked >= max_checks:
                break

            pair = tuple(sorted([str(seed["id"]), str(cand["id"])]))
            if pair in tried:
                continue
            tried.add(pair)

            # Different source → potentially same claim, different conclusions
            src_a = seed["payload"].get("source_l0") or seed["payload"].get("source") or ""
            src_b = cand["payload"].get("source_l0") or cand["payload"].get("source") or ""
            if src_a and src_b and src_a == src_b:
                continue  # same source → probably complementary, not contradictory

            checked += 1
            analysis = _check_contradiction(seed, cand)
            if not analysis:
                continue

            if not analysis.get("is_contradictory"):
                continue

            contradictions += 1
            confidence = analysis.get("confidence", 0.5)
            ctype = analysis.get("contradiction_type", "direct")

            LOG.info(
                f"Contradiction found ({ctype}, conf={confidence:.2f}): "
                f"'{seed['payload'].get('title')}' vs '{cand['payload'].get('title')}'"
            )

            if confidence >= 0.85 and ctype == "direct":
                # High confidence direct contradiction — prefer the one with higher completeness
                score_a = seed["payload"].get("completeness_score", 0)
                score_b = cand["payload"].get("completeness_score", 0)
                if score_a > score_b + 0.1:
                    qc.patch_payload(cand["id"], {
                        "status": "superseded",
                        "superseded_reason": f"矛盾检测：与 {seed['payload'].get('title')} 矛盾，完整度较低",
                    })
                    LOG.info(f"Auto-superseded: {cand['payload'].get('title')}")
                    continue
                # Not clear-cut → write to inbox

            # Write to inbox for user decision
            inbox_path = _write_inbox_contradiction(seed, cand, analysis)
            inbox_written += 1
            LOG.info(f"Contradiction written to inbox: {inbox_path}")

            # Update pending decisions count
            ctx = sm.load_context()
            ctx["pending_user_decisions"] = ctx.get("pending_user_decisions", 0) + 1
            sm.save_context(ctx)

    LOG.info(f"Contradiction detection done: checked={checked} contradictions={contradictions} inbox={inbox_written}")
    return {"checked": checked, "contradictions": contradictions, "inbox_written": inbox_written}
