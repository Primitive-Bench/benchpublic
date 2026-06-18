"""Slice taxonomy + deterministic assignment + per-slice wins matrix.

Ported from arlenk2021/GoldenEvalsWebSearch `src/slices.py` (the taxonomy +
assignment) and `src/reporting/slices.py` (the per-slice leaderboard + wins
matrix). Tie bands are delegated to `bench_stats.tied_rank_band` (the shared
"name a winner only when its Wilson interval clears the runner-up's" decision),
and per-proportion CIs to `bench_stats.wilson`.

A **slice** is a cross-cutting tag describing a query's intent/domain — orthogonal
to `stratum` (difficulty kind) and `source` (registry). Slices turn one
leaderboard into many: "Exa wins company_lookup", "exa/serpapi tie
government_registry", "Brave wins broad web freshness". A row may carry several
slices.

Slices are part of the frozen row contract: assigned deterministically from the
row's source + metadata (no model, no fetch), so they are reproducible and
auditable. To add a slice, add a source that produces it (see SOURCE_SLICES) or a
content rule below — then re-run the reslice step.
"""

from __future__ import annotations

import re
from collections import defaultdict

from bench_stats import tied_rank_band, wilson

# Canonical slice vocabulary (the 10 intent slices; maps the Stage-2 product list).
SLICES = (
    "company_lookup",  # startup/company lookup
    "government_registry",  # gov/registry pages
    "citation_needed",  # claims needing an authoritative citation
    "technical_docs",  # technical documentation / specs
    "fresh_news",  # time-sensitive news/events
    "pricing_pages",  # pricing / plans pages
    "b2b_tools",  # obscure B2B tools / vendors
    "docs_lookup",  # general docs lookup
    "navigational",  # direct, known-item navigational lookups
    "long_tail",  # rare, low-traffic queries
)

# Per-source default slices. The current registries are all authoritative
# institutional documents, but they map to distinct *intents*: SEC filings ARE a
# company lookup; NVD records ARE technical/security docs; FR documents are the
# canonical citation-needed government source.
SOURCE_SLICES: dict[str, tuple[str, ...]] = {
    "fed_register": ("government_registry", "citation_needed"),
    "sec_edgar": ("government_registry", "company_lookup", "citation_needed"),
    "nvd_cve": ("government_registry", "technical_docs", "citation_needed"),
    "github_releases": ("technical_docs", "docs_lookup"),
    "sentinel": (),  # sentinels measure index freshness, not an intent slice
}

# Lightweight content rules (additive, deterministic). Keyword hits on the query
# text add a slice. Kept conservative so assignment stays auditable.
_CONTENT_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("pricing_pages", re.compile(r"\b(pricing|plans?|per month|/mo|subscription cost)\b", re.I)),
    ("fresh_news", re.compile(r"\b(today|breaking|announced|this week|just released)\b", re.I)),
    ("b2b_tools", re.compile(r"\b(API|SDK|integration|webhook|SaaS platform)\b")),
    ("navigational", re.compile(r"\b(official site|homepage|login page|sign in to)\b", re.I)),
)

# A title shorter than this many chars with a rare proper noun is a long-tail proxy.
_LONG_TAIL_MAXLEN = 60


def assign_slices(row: dict) -> list[str]:
    """Deterministic slice tags for a row. Union of source defaults + content rules."""
    tags: set[str] = set(SOURCE_SLICES.get(row.get("source", ""), ()))
    text = " ".join(
        filter(
            None,
            [
                row.get("query", ""),
                *(row.get("query_variants", {}) or {}).values(),
            ],
        )
    )
    for slice_name, pattern in _CONTENT_RULES:
        if pattern.search(text):
            tags.add(slice_name)
    # long-tail proxy: a short, specific descriptive query
    if 0 < len(row.get("query", "")) <= _LONG_TAIL_MAXLEN:
        tags.add("long_tail")
    return sorted(tags)


# ---------------------------------------------------------------------------
# Per-slice leaderboards + a slice->winner matrix.
#
# Turns one ranking into many: for each slice (query intent), rank vendors and
# call a winner only when the top Wilson interval clears the runner-up's —
# otherwise a TIE band. This is what lets the benchmark say "Exa wins
# company_lookup, exa/serpapi tie government_registry" instead of "Vendor A is #1".
# ---------------------------------------------------------------------------
MIN_SLICE_N = 8  # below this a slice cell prints n but is flagged thin


def slice_rows(meta: dict[str, dict]) -> dict[str, set[str]]:
    """{slice -> set of row_ids} from a {row_id -> row metadata} map."""
    rows: dict[str, set[str]] = defaultdict(set)
    for rid, m in meta.items():
        for s in m.get("slices", []):
            rows[s].add(rid)
    return rows


def slice_leaderboard(
    rids: set[str], vendors: list[str], outcomes: dict[tuple[str, str], int]
) -> list[tuple[str, float, float, float, int]]:
    """Per-vendor (name, point, low, high, n) Wilson rates over the slice's rows.

    `outcomes` is {(row_id, vendor): hit} (best-over-reps). A vendor is only
    rated on the rows it actually covered, so a skipped vendor is not charged a
    miss it never had a chance at.
    """
    rates: list[tuple[str, float, float, float, int]] = []
    for v in vendors:
        covered = [rid for rid in rids if (rid, v) in outcomes]
        if not covered:
            continue
        hits = sum(outcomes[(rid, v)] for rid in covered)
        ci = wilson(
            hits, len(covered)
        )  # bench_schemas.StatTest: .statistic=point, .ci_low/.ci_high
        rates.append((v, ci.statistic, ci.ci_low, ci.ci_high, len(covered)))
    rates.sort(key=lambda x: -x[1])
    return rates


def winner_band(rates: list[tuple[str, float, float, float, int]]) -> list[str]:
    """Tie band = vendors whose Wilson interval overlaps the leader's.

    Delegates to bench_stats.tied_rank_band (shared separability decision); the
    trailing per-vendor n is dropped before handing off the (name, point, low,
    high) tuples it expects.
    """
    if not rates:
        return []
    return tied_rank_band([(v, p, lo, hi) for v, p, lo, hi, _ in rates]).band


def wins_matrix(meta: dict[str, dict], outcomes: dict[tuple[str, str], int]) -> list[dict]:
    """The slice->winner matrix. One entry per slice with data:

        {slice, n, band, winner, losers, rates}

    `winner` is set only when the band is a single vendor; otherwise it is None
    and `band` is the tie group. `losers` are vendors that covered the slice but
    fell out of the band ("loses here").
    """
    vendors = sorted({v for (_, v) in outcomes})
    srows = slice_rows(meta)
    matrix: list[dict] = []
    for s in SLICES:
        rids = srows.get(s, set())
        if not rids:
            continue
        rates = slice_leaderboard(rids, vendors, outcomes)
        if not rates:
            continue
        band = winner_band(rates)
        n = max(r[4] for r in rates)
        losers = [v for v in vendors if v not in band and any((rid, v) in outcomes for rid in rids)]
        matrix.append(
            {
                "slice": s,
                "n": n,
                "band": band,
                "winner": band[0] if len(band) == 1 else None,
                "losers": losers,
                "rates": rates,
            }
        )
    return matrix
