#!/usr/bin/env python3
"""ATLAS 自主知识代理 v1.2 (Phase 3) — 主循环"""
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from core.perception import perceive
from core.decision import decide
from core.reflection import reflect
from core.scheduler import should_run_deep, should_run_evolution, mark_deep_done, mark_evolution_done
from storage import state_manager as sm
from communication.report_writer import write_daily_report, write_evolution_report

LOG = setup_logger("atlas", log_dir="/Users/luolimo/Library/Logs")

FAST_INTERVAL_SECONDS = 15 * 60
_shutdown = False

_weekly_stats: dict = {
    "nodes_created":  0,
    "l1_completed":   0,
    "l2_generated":   0,
    "l3_synthesized": 0,
}


def _handle_signal(sig, frame):
    global _shutdown
    LOG.info(f"Received signal {sig}, shutting down gracefully...")
    _shutdown = True


def _execute_task(task) -> dict:
    result: dict = {"type": task.type, "error": None}
    ctx = task.context

    try:
        if task.type == "intake":
            from actions.knowledge.intake import run_intake
            r = run_intake()
            result.update(r)

        elif task.type == "l0_organize":
            from actions.knowledge.organize import run_l0_organize
            r = run_l0_organize(max_per_cycle=ctx.get("max", 10))
            result.update(r)
            _weekly_stats["nodes_created"] += r.get("nodes_created", 0)

        elif task.type == "l1_complete":
            from actions.knowledge.complete import run_l1_complete
            r = run_l1_complete(max_per_cycle=ctx.get("max", 5))
            result.update(r)
            _weekly_stats["l1_completed"] += r.get("completed", 0)

        elif task.type == "l2_generate":
            from actions.knowledge.associate import run_l2_generate
            r = run_l2_generate(max_per_cycle=ctx.get("max", 3))
            result.update(r)
            _weekly_stats["l2_generated"] += r.get("generated", 0)

        elif task.type == "l3_synthesize":
            from actions.knowledge.synthesize import run_l3_synthesize
            r = run_l3_synthesize(max_per_cycle=ctx.get("max", 1))
            result.update(r)
            _weekly_stats["l3_synthesized"] += r.get("synthesized", 0)

        elif task.type == "l1_consolidate":
            from actions.knowledge.consolidate import run_l1_consolidate
            r = run_l1_consolidate(max_per_cycle=ctx.get("max", 3))
            result.update(r)

        elif task.type == "data_quality":
            from actions.maintenance.data_quality import run_data_quality
            r = run_data_quality(evolve=task.context.get("evolve", False))
            result.update(r)

        elif task.type == "contradiction_detect":
            from actions.maintenance.contradiction import run_contradiction_detection
            r = run_contradiction_detection(max_checks=5)
            result.update(r)

        elif task.type == "autonomous_research":
            from actions.maintenance.research import run_autonomous_research
            r = run_autonomous_research(max_domains=1)
            result.update(r)

        elif task.type == "process_user_response":
            from communication.inbox_monitor import process_user_responses
            r = process_user_responses()
            result.update(r)

        elif task.type == "process_external_message":
            from communication.agent_protocol import process_external_messages
            r = process_external_messages()
            result.update(r)

    except Exception as e:
        LOG.error(f"Task {task.type} failed: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def run_cycle() -> None:
    cycle_start_ms = int(time.time() * 1000)

    # Always process inbox + external messages first (cheap, high priority)
    try:
        from communication.inbox_monitor import process_user_responses
        process_user_responses()
    except Exception as e:
        LOG.error(f"Inbox monitor failed: {e}")

    try:
        from communication.agent_protocol import process_external_messages
        process_external_messages()
    except Exception as e:
        LOG.error(f"External messages failed: {e}")

    ws    = perceive()
    tasks = decide(ws, max_tasks=4)

    cycle_results = []
    errors        = []

    for task in tasks:
        if task.type in ("write_daily_report", "process_user_response", "process_external_message"):
            continue
        LOG.info(f"Executing: {task.type} (priority={task.priority:.2f}) — {task.description}")
        r = _execute_task(task)
        cycle_results.append(r)
        if r.get("error"):
            errors.append(f"{task.type}: {r['error']}")

    # Write daily report once per day
    today_report = (
        Path("/Volumes/data/obsidian-vault/agent-brain/daily")
        / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    )
    world_summary = {
        "knowledge_health": {
            "pending_l0":    ws.pending_l0,
            "l1_total":      ws.l1_total,
            "incomplete_l1": ws.incomplete_l1,
            "l2_total":      ws.l2_total,
            "l3_total":      ws.l3_total,
            "l4_total":      ws.l4_total,
        },
        "knowledge_gaps": ws.knowledge_gaps,
    }
    if not today_report.exists():
        write_daily_report(cycle_results, world_summary)

    ctx = sm.load_context()
    ctx["knowledge_health"] = world_summary["knowledge_health"]
    ctx["knowledge_gaps"]   = ws.knowledge_gaps
    sm.save_context(ctx)

    # Sync fresh ATLAS state into Claude Code memory
    try:
        from communication.claude_memory_sync import sync_to_claude_memory
        sync_to_claude_memory()
    except Exception as e:
        LOG.warning(f"Claude memory sync failed: {e}")

    duration_ms = int(time.time() * 1000) - cycle_start_ms
    reflect(ws.cycle_id, duration_ms, cycle_results, errors)


def run_deep_cycle() -> None:
    LOG.info("=== DEEP CYCLE START ===")
    ws = perceive()
    world_summary = {
        "knowledge_health": {
            "pending_l0":    ws.pending_l0,
            "l1_total":      ws.l1_total,
            "incomplete_l1": ws.incomplete_l1,
            "l2_total":      ws.l2_total,
            "l3_total":      ws.l3_total,
            "l4_total":      ws.l4_total,
        },
        "knowledge_gaps": ws.knowledge_gaps,
    }
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_report = Path(f"/Volumes/data/obsidian-vault/agent-brain/daily/{today}.md")
    if today_report.exists():
        today_report.unlink()
    write_daily_report([], world_summary)
    mark_deep_done()
    LOG.info("=== DEEP CYCLE DONE ===")


def run_evolution_cycle() -> None:
    LOG.info("=== EVOLUTION CYCLE START ===")

    # 1. Health assessment
    try:
        from core.health import assess_health
        health = assess_health()
    except Exception as e:
        LOG.error(f"Health assessment failed: {e}")
        health = None

    # 2. Self-model update
    try:
        from actions.evolution.self_model import update_self_model
        update_self_model()
    except Exception as e:
        LOG.error(f"Self-model update failed: {e}")

    # 3. L4 → L1 feedback
    l4_results = {"l4_processed": 0, "l1_enriched": 0}
    try:
        from actions.evolution.l4_feedback import run_l4_feedback
        l4_results = run_l4_feedback(max_l4=3)
    except Exception as e:
        LOG.error(f"L4 feedback failed: {e}")

    # 4. L4 Obsidian sync (backfill any L4 records missing files)
    try:
        from actions.evolution.l4_obsidian import sync_l4_obsidian
        sync_l4_obsidian()
    except Exception as e:
        LOG.error(f"L4 Obsidian sync failed: {e}")

    # 4b. Outbox archive — roll up previous day/week/month files
    try:
        from communication.outbox_archiver import run_archiver
        ar = run_archiver()
        if any(ar.values()):
            LOG.info(f"Outbox archived: {ar}")
    except Exception as e:
        LOG.error(f"Outbox archiver failed: {e}")

    # 5. Prompt optimization
    prompt_results = {"evaluated": 0, "upgraded": 0}
    try:
        from actions.evolution.prompt_optimizer import run_prompt_optimization
        prompt_results = run_prompt_optimization()
    except Exception as e:
        LOG.error(f"Prompt optimization failed: {e}")

    # 6. Strategy adjustment
    strategy_results: dict = {}
    try:
        from actions.evolution.strategy_adjuster import adjust_strategy
        if health:
            strategy_results = adjust_strategy(health)
            # Cache health grade in strategy file for decision.py
            from actions.evolution.strategy_adjuster import _load_strategy, _save_strategy
            s = _load_strategy()
            s["health_grade"] = getattr(health, "grade", "?")
            _save_strategy(s)
    except Exception as e:
        LOG.error(f"Strategy adjustment failed: {e}")

    # 7. Write evolution report
    try:
        self_model = sm.load_self_model()
        write_evolution_report(
            _weekly_stats.copy(),
            health,
            self_model,
            prompt_results,
            strategy_results,
            l4_results,
        )
    except Exception as e:
        LOG.error(f"Evolution report failed: {e}")

    # 8. Reset weekly stats
    for k in _weekly_stats:
        _weekly_stats[k] = 0

    mark_evolution_done()
    LOG.info("=== EVOLUTION CYCLE DONE ===")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    LOG.info("=" * 60)
    LOG.info("ATLAS 自主知识代理 v1.2 (Phase 3) 启动")
    LOG.info("=" * 60)

    sm._ensure()

    # Start MCP query server in background
    try:
        from communication.mcp_server import start_server
        start_server()
        LOG.info("MCP server started on http://127.0.0.1:8766")
    except Exception as e:
        LOG.warning(f"MCP server failed to start: {e}")

    cycle_count = 0

    while not _shutdown:
        cycle_start = time.time()
        cycle_count += 1
        LOG.info(f"--- Fast Cycle #{cycle_count} ---")

        try:
            run_cycle()
        except Exception as e:
            LOG.error(f"Cycle error: {e}", exc_info=True)

        try:
            if should_run_deep():
                run_deep_cycle()
        except Exception as e:
            LOG.error(f"Deep cycle error: {e}", exc_info=True)

        try:
            if should_run_evolution():
                run_evolution_cycle()
        except Exception as e:
            LOG.error(f"Evolution cycle error: {e}", exc_info=True)

        # Manual evolution trigger: touch agent-brain/state/trigger_evolution.flag
        try:
            trigger_flag = Path("/Volumes/data/obsidian-vault/agent-brain/state/trigger_evolution.flag")
            if trigger_flag.exists():
                LOG.info("Manual evolution trigger detected — running evolution cycle now")
                trigger_flag.unlink()
                run_evolution_cycle()
        except Exception as e:
            LOG.error(f"Manual trigger error: {e}", exc_info=True)

        # Todo agent processing
        try:
            from actions.todo_agent import process_todo_tasks
            process_todo_tasks()
        except Exception as e:
            LOG.error(f"Todo agent error: {e}", exc_info=True)

        if _shutdown:
            break

        elapsed = time.time() - cycle_start
        sleep_s = max(0, FAST_INTERVAL_SECONDS - elapsed)
        LOG.info(f"Cycle done in {elapsed:.1f}s. Next cycle in {sleep_s/60:.1f} min.")

        for _ in range(int(sleep_s)):
            if _shutdown:
                break
            time.sleep(1)

    LOG.info("ATLAS 代理 已停止")


if __name__ == "__main__":
    main()
