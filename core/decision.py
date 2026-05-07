"""Decision layer — priority matrix → top-N task list for this cycle.

Reads strategy.json overrides (written by strategy_adjuster.py).
Uses self-model to boost/reduce domain-specific research tasks.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from core.perception import WorldState

LOG = logging.getLogger("atlas.decision")

STRATEGY_FILE = Path("/Volumes/data/obsidian-vault/agent-brain/state/strategy.json")


@dataclass
class Task:
    type: str
    priority: float
    context: dict
    description: str


# Base (urgency, importance, user_relevance)
_BASE = {
    "process_external_message": (1.0, 0.8, 0.8),
    "process_user_response":    (1.0, 0.9, 1.0),
    "intake":                   (0.9, 0.9, 0.7),  # file intake: high priority, cheap
    "l0_organize":              (0.8, 0.9, 0.6),
    "l1_complete":              (0.4, 0.6, 0.4),
    "l1_consolidate":           (0.5, 0.8, 0.7),  # high importance: fixes fragmentation
    "l2_generate":              (0.3, 0.7, 0.5),
    "l3_synthesize":            (0.3, 0.8, 0.4),
    "contradiction_detect":     (0.4, 0.7, 0.5),
    "autonomous_research":      (0.2, 0.8, 0.6),
    "write_daily_report":       (0.5, 0.7, 0.9),
    "data_quality":             (0.3, 0.9, 0.3),  # high importance: keeps KB clean
}


def _score(u: float, i: float, r: float) -> float:
    return u * 0.4 + i * 0.4 + r * 0.2


def _dynamic_max(backlog: int, *, base: int, cap: int, override: int | None = None) -> int:
    """Scale per-cycle batch size with backlog depth.

    Tiers (relative to base):
      backlog ≤ 10          → base          (steady state)
      backlog 11-50         → base × 2      (mild pressure)
      backlog 51-200        → base × 4      (heavy pressure)
      backlog > 200         → cap           (max throughput)

    strategy.json override (if set) always wins.
    """
    if override is not None:
        return override
    if backlog <= 10:
        return base
    if backlog <= 50:
        return min(cap, base * 2)
    if backlog <= 200:
        return min(cap, base * 4)
    return cap


def _load_strategy() -> dict:
    if STRATEGY_FILE.exists():
        try:
            import json
            return json.loads(STRATEGY_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _cycle_counter() -> int:
    """Approximate cycle count from cycle-log length."""
    log = Path("/Volumes/data/obsidian-vault/agent-brain/state/cycle-log.jsonl")
    if not log.exists():
        return 0
    try:
        return sum(1 for _ in log.open("utf-8"))
    except Exception:
        return 0


def decide(ws: WorldState, max_tasks: int = 4) -> list[Task]:
    strategy = _load_strategy()
    overrides = strategy.get("priority_overrides", {})
    cycle_n   = _cycle_counter()

    tasks: list[Task] = []

    if ws.external_messages > 0:
        u, i, r = _BASE["process_external_message"]
        tasks.append(Task("process_external_message", _score(u, i, r),
                           {"count": ws.external_messages}, "处理外部 Agent 消息"))

    if ws.user_responses > 0:
        u, i, r = _BASE["process_user_response"]
        tasks.append(Task("process_user_response", _score(u, i, r),
                           {"count": ws.user_responses}, "处理用户收件箱回复"))

    # File intake: always check every cycle (fast, no LLM unless domain unknown)
    u, i, r = _BASE["intake"]
    tasks.append(Task("intake", _score(u, i, r), {}, "扫描 intake/ 文件夹入库"))

    if ws.pending_l0 > 0:
        u, i, r = _BASE["l0_organize"]
        max_l0 = _dynamic_max(
            ws.pending_l0, base=10, cap=40,
            override=strategy.get("max_l0_per_cycle"),
        )
        tasks.append(Task("l0_organize", _score(u, i, r),
                           {"pending": ws.pending_l0, "max": max_l0},
                           f"处理 {ws.pending_l0} 条 L0 原料 → L1 (batch={max_l0})"))

    if ws.incomplete_l1 > 0:
        u, i, r = _BASE["l1_complete"]
        base_p = _score(u, i, r)
        p = overrides.get("l1_complete", base_p)
        max_l1c = _dynamic_max(
            ws.incomplete_l1, base=8, cap=40,
            override=strategy.get("max_l1_complete_per_cycle"),
        )
        tasks.append(Task("l1_complete", p,
                           {"count": ws.incomplete_l1, "max": max_l1c},
                           f"补全 {ws.incomplete_l1} 条不完整 L1 (batch={max_l1c})"))

    if ws.l1_total > 20:
        u, i, r = _BASE["l2_generate"]
        base_p = _score(u, i, r)
        p = overrides.get("l2_generate", base_p)
        max_l2 = strategy.get("max_l2_per_cycle", 3)
        tasks.append(Task("l2_generate", p,
                           {"l1_total": ws.l1_total, "max": max_l2},
                           f"生成 L2 跨域洞见 (batch={max_l2})"))

    max_l3 = strategy.get("max_l3_per_cycle", 1)
    if ws.l2_total > 15 and max_l3 > 0:
        u, i, r = _BASE["l3_synthesize"]
        tasks.append(Task("l3_synthesize", _score(u, i, r),
                           {"l2_total": ws.l2_total, "max": max_l3},
                           "蒸馏 L3 框架"))

    # Consolidation: run every 2 cycles while generic-titled fragments exist
    if ws.l1_total > 30 and cycle_n % 2 == 0:
        u, i, r = _BASE["l1_consolidate"]
        tasks.append(Task("l1_consolidate", _score(u, i, r),
                           {"max": 3}, "合并碎片化 L1 节点"))

    # Data quality: run every cycle when issues exist, else every 3 cycles
    quality_issues = ws.unclassified_count + ws.no_category_count + ws.garbled_count
    quality_every  = 1 if quality_issues > 0 else 3
    if cycle_n % quality_every == 0:
        u, i, r = _BASE["data_quality"]
        # Urgency scales with issue count
        urgency = min(1.0, 0.3 + quality_issues / 200)
        evolve  = cycle_n % strategy.get("taxonomy_evolve_every_n_cycles", 10) == 0
        tasks.append(Task("data_quality", _score(urgency, i, r),
                          {"quality_issues": quality_issues, "evolve": evolve},
                          f"数据质量维护 (未分类={ws.unclassified_count} "
                          f"无大类={ws.no_category_count} 乱码={ws.garbled_count})"))

    # Contradiction detection: run every N cycles per strategy
    check_every = strategy.get("contradiction_check_every_n_cycles", 4)
    if ws.l1_total > 50 and cycle_n % check_every == 0:
        u, i, r = _BASE["contradiction_detect"]
        tasks.append(Task("contradiction_detect", _score(u, i, r) * 0.65,
                           {}, "矛盾检测"))

    # Research: run every N cycles per strategy (with cooldown)
    research_every = strategy.get("research_cooldown_cycles", 3)
    if ws.knowledge_gaps and cycle_n % research_every == 0:
        u, i, r = _BASE["autonomous_research"]
        base_p = _score(u, i, r)
        p = overrides.get("autonomous_research", base_p)
        tasks.append(Task("autonomous_research", p,
                           {"gaps": ws.knowledge_gaps[:3]},
                           f"自主研究：{ws.knowledge_gaps[0]}"))

    tasks.sort(key=lambda t: t.priority, reverse=True)
    selected = tasks[:max_tasks]

    LOG.info(f"Decided {len(selected)} tasks: {[t.type for t in selected]} "
             f"(cycle #{cycle_n}, grade={strategy.get('health_grade', '?')})")
    return selected
