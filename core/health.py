"""Knowledge network health assessment.

Computes a comprehensive health report of the knowledge base:
- Coverage: what % of L1 nodes have L2 cross-domain connections
- Orphan rate: L1 nodes with no L2 links
- Freshness: avg age of knowledge
- Domain balance: distribution evenness
- Pyramid health: L4→L3→L2→L1 ratio analysis
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from storage import qdrant_client as qc

LOG = logging.getLogger("atlas.health")


@dataclass
class HealthReport:
    timestamp: str = ""

    # Counts
    l1_total: int = 0
    l2_total: int = 0
    l3_total: int = 0
    l4_total: int = 0

    # Coverage
    l1_with_l2: int = 0           # L1 nodes referenced by at least one L2
    coverage_score: float = 0.0   # l1_with_l2 / l1_total

    # Completeness
    avg_l1_completeness: float = 0.0
    l1_incomplete_count: int = 0

    # Domain health
    domain_count: int = 0
    top_domain: str = ""
    top_domain_l1: int = 0
    gap_domains: list[str] = field(default_factory=list)  # domains with <5 L1

    # Pyramid ratios (healthy targets in parentheses)
    l2_per_l1: float = 0.0        # target ~0.3
    l3_per_l2: float = 0.0        # target ~0.05
    l4_per_l3: float = 0.0        # target ~0.02

    # Overall health score (0-1)
    score: float = 0.0
    grade: str = ""               # A/B/C/D


def _compute_l1_with_l2_coverage(l1_pts: list[dict], l2_pts: list[dict]) -> int:
    """Count L1 nodes referenced by at least one L2.

    Supports two payload schemas:
      - Old plugin format: source_ids (list of int/str)
      - New agent format:  source_id_a, source_id_b (individual str)
    """
    referenced: set[str] = set()
    for l2 in l2_pts:
        pay = l2["payload"]
        # New agent format
        for field in ("source_id_a", "source_id_b"):
            v = pay.get(field)
            if v:
                referenced.add(str(v))
        # Old plugin format
        for sid in pay.get("source_ids") or []:
            referenced.add(str(sid))
    return sum(1 for p in l1_pts if str(p["id"]) in referenced)


def assess_health() -> HealthReport:
    report = HealthReport(timestamp=datetime.now(timezone.utc).isoformat())

    # Fetch all active records
    l1_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=2000)

    l2_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 2}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=2000)

    l3_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 3}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=500)

    l4_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 4}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=100)

    report.l1_total = len(l1_pts)
    report.l2_total = len(l2_pts)
    report.l3_total = len(l3_pts)
    report.l4_total = len(l4_pts)

    # Coverage
    report.l1_with_l2    = _compute_l1_with_l2_coverage(l1_pts, l2_pts)
    report.coverage_score = report.l1_with_l2 / report.l1_total if report.l1_total > 0 else 0

    # Completeness
    total_score = sum(p["payload"].get("completeness_score", 0) for p in l1_pts)
    report.avg_l1_completeness = total_score / report.l1_total if report.l1_total > 0 else 0
    report.l1_incomplete_count = sum(
        1 for p in l1_pts if p["payload"].get("completeness_score", 0) < 0.85
    )

    # Domain analysis
    domain_counts: dict[str, int] = defaultdict(int)
    for pt in l1_pts:
        d = pt["payload"].get("domain") or "未分类"
        domain_counts[d] += 1

    report.domain_count = len(domain_counts)
    if domain_counts:
        top = max(domain_counts.items(), key=lambda x: x[1])
        report.top_domain    = top[0]
        report.top_domain_l1 = top[1]
    report.gap_domains = [d for d, c in domain_counts.items() if c < 5]

    # Pyramid ratios
    report.l2_per_l1 = report.l2_total / report.l1_total if report.l1_total > 0 else 0
    report.l3_per_l2 = report.l3_total / report.l2_total if report.l2_total > 0 else 0
    report.l4_per_l3 = report.l4_total / report.l3_total if report.l3_total > 0 else 0

    # Overall health score
    components = {
        "completeness":  min(report.avg_l1_completeness / 0.85, 1.0) * 0.30,
        "coverage":      min(report.coverage_score / 0.30, 1.0) * 0.25,
        "pyramid":       min(report.l2_per_l1 / 0.25, 1.0) * 0.20,
        "domain_depth":  min(1 - len(report.gap_domains) / max(report.domain_count, 1), 1.0) * 0.15,
        "l3_l4_exist":   (1.0 if report.l3_total > 5 else 0.5) * 0.10,
    }
    report.score = sum(components.values())

    if report.score >= 0.85:
        report.grade = "A"
    elif report.score >= 0.70:
        report.grade = "B"
    elif report.score >= 0.55:
        report.grade = "C"
    else:
        report.grade = "D"

    LOG.info(
        f"Health: score={report.score:.3f} ({report.grade}) | "
        f"L1={report.l1_total} L2={report.l2_total} L3={report.l3_total} L4={report.l4_total} | "
        f"coverage={report.coverage_score:.2%} completeness={report.avg_l1_completeness:.2%}"
    )
    return report
