"""Verification stage — the "truth checked at use" liveness gate.

For each candidate:
  1. Reject if the query text leaks the golden URL (rule 2).
  2. Fetch the golden URL; assert the truth token appears in MAIN content.
  3. Build the equivalence class: the canonical, plus the post-redirect final
     URL and any <link rel=canonical>, admitted ONLY when on the same
     registrable domain AND the token is present.
  4. Require TWO passing fetches at least VERIFY_MIN_GAP_HOURS apart before a
     candidate is promoted to a golden row (truth checked at use, not just at
     birth). Pass --single-pass for fast local integration tests.

Every rejection is logged with a reason. Nothing is silently dropped.

`liveness_gate(cand)` is the single-candidate gate: it returns a `Liveness`
record carrying whether the candidate is currently live (truth-token present in
main content on an authoritative fetch) and the equivalence-class payload.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from bench_core.config import get_settings
from bench_core.domain import GoldenRow
from bench_core.http import fetch
from bench_core.storage import GOLDEN, read_jsonl, write_jsonl
from bench_core.storage import CANDIDATES, REJECTIONS, log_rejection
from bench_core.urls import EquivalenceClass, normalize, same_registrable_domain

CANDIDATES_FILE = CANDIDATES / "candidates.jsonl"
GOLDEN_FILE = GOLDEN / "golden.jsonl"
VERIFICATIONS_FILE = REJECTIONS.parent / "verifications.jsonl"  # passing-fetch ledger


# ---------------------------------------------------------------------------
# Slice taxonomy + deterministic assignment.
#
# A **slice** is a cross-cutting tag describing a query's intent/domain —
# orthogonal to `stratum` (difficulty kind) and `source` (registry). Slices turn
# one leaderboard into many. A row may carry several slices. Slices are part of
# the frozen row contract: assigned deterministically from the row's source +
# metadata (no model, no fetch), so they are reproducible and auditable.
# ---------------------------------------------------------------------------

# Canonical slice vocabulary (maps the Stage-2 product list).
SLICES = (
    "government_registry",  # gov/registry pages
    "company_lookup",  # startup/company lookup
    "technical_docs",  # technical documentation / specs
    "docs_lookup",  # general docs lookup
    "fresh_news",  # time-sensitive news/events
    "b2b_tools",  # obscure B2B tools / vendors
    "local_regional",  # local / regional sources
    "pricing_pages",  # pricing / plans pages
    "long_tail",  # rare, low-traffic queries
    "citation_needed",  # claims needing an authoritative citation
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
    ("local_regional", re.compile(r"\b(near me|in [A-Z][a-z]+ (county|city)|state of)\b")),
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
# Liveness gate
# ---------------------------------------------------------------------------
@dataclass
class Liveness:
    """Result of the per-candidate liveness gate ("truth checked at use").

    `live` is True iff an authoritative fetch currently shows the truth token in
    MAIN content. `reason` is the machine-readable verdict ("ok" or a rejection
    code). `payload` carries the verified equivalence-class members + token depth
    when live, else None.
    """

    live: bool
    reason: str
    payload: dict | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_leaks_url(query: str, golden_url: str) -> bool:
    norm = normalize(golden_url)
    bare = norm.split("://", 1)[-1]
    q = query.lower()
    return golden_url.lower() in q or bare.lower() in q


def mirror_candidates(cand: dict) -> list[str]:
    """Known same-domain alias URLs by source, to be VERIFIED before admission.

    Federal Register exposes a stable short link www.federalregister.gov/d/{docnum}
    that 301s to the canonical document — a real mirror. Extend per source as
    verified aliases are found (SEC viewer URLs, etc.). Returns [] for sources
    with no known alias, leaving the equivalence class at size 1 honestly.
    """
    src = cand.get("source")
    token = cand["truth_token"]
    if src == "fed_register":
        return [f"https://www.federalregister.gov/d/{token}"]
    return []


async def liveness_gate(cand: dict) -> Liveness:
    """Single-candidate gate: fetch the golden URL and check the truth token is
    live in MAIN content, building the verified equivalence class.

    Returns a `Liveness` record. This is the "truth checked at use" gate — the
    same check is applied at promotion time AND can be re-run at probe time to
    confirm the fact is still live before a vendor is scored against it.
    """
    token = cand["truth_token"]
    res = await fetch(cand["golden_url"])
    if res.status == 0:
        return Liveness(False, "fetch_error", None)
    if res.soft_404:
        return Liveness(False, "soft_404", None)
    if res.status >= 400:
        return Liveness(False, f"http_{res.status}", None)
    if token not in res.main_text:
        return Liveness(False, "truth_token_not_in_main_content", None)

    eq = EquivalenceClass(cand["golden_url"])
    # post-redirect final URL joins only if same registrable domain + token present
    if same_registrable_domain(cand["golden_url"], res.final_url):
        eq.add(res.final_url)
    # declared canonical joins only if same registrable domain (token already verified)
    if res.canonical_link and same_registrable_domain(cand["golden_url"], res.canonical_link):
        eq.add(res.canonical_link)
    # source-specific known mirrors: admitted ONLY after passing the same
    # token + same-registrable-domain check (no blind aliasing).
    for mirror in mirror_candidates(cand):
        if not same_registrable_domain(cand["golden_url"], mirror):
            continue
        m = await fetch(mirror)
        if m.status == 200 and not m.soft_404 and token in m.main_text:
            eq.add(mirror)
            if same_registrable_domain(cand["golden_url"], m.final_url):
                eq.add(m.final_url)
    # canonical token depth (offset in the source-of-truth main content) lets the
    # eval stratify by where the token sits — title-zone vs deep-body — so a
    # snippet-only extractor cannot look good on title-token sources.
    depth = res.main_text.find(token)
    return Liveness(
        True,
        "ok",
        {
            "canonical": eq.canonical,
            "members": eq.members,
            "token_depth": depth,
            "canonical_chars": len(res.main_text),
        },
    )


async def _verify_one(cand: dict) -> tuple[bool, str, dict | None]:
    """Returns (passed, reason, equivalence_payload). Thin wrapper over the
    liveness gate, preserved for the run loop below."""
    result = await liveness_gate(cand)
    return result.live, result.reason, result.payload


def _passing_ledger() -> dict[str, list[str]]:
    ledger: dict[str, list[str]] = {}
    for rec in read_jsonl(VERIFICATIONS_FILE):
        ledger.setdefault(rec["row_id"], []).append(rec["ts"])
    return ledger


async def run(single_pass: bool = False, limit: int | None = None) -> int:
    s = get_settings()
    candidates = list(read_jsonl(CANDIDATES_FILE))
    if limit:
        candidates = candidates[:limit]
    already_golden = {r["row_id"] for r in read_jsonl(GOLDEN_FILE)}
    ledger = _passing_ledger()
    promoted = 0
    new_goldens: list[dict] = []

    for cand in candidates:
        rid = cand["row_id"]
        if rid in already_golden:
            continue
        if query_leaks_url(cand["query"], cand["golden_url"]):
            log_rejection(rid, "query_leaks_golden_url", cand["query"], ts=_utcnow_iso())
            continue

        passed, reason, payload = await _verify_one(cand)
        now = _utcnow_iso()
        if not passed:
            log_rejection(rid, reason, cand["golden_url"], ts=now)
            continue

        # record this passing fetch
        write_jsonl(
            VERIFICATIONS_FILE,
            [{"row_id": rid, "ts": now, "members": payload["members"]}],
            append=True,
        )
        times = ledger.setdefault(rid, [])
        times.append(now)

        if not single_pass:
            if len(times) < 2:
                continue
            gap = (
                abs(
                    datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])
                ).total_seconds()
                / 3600.0
            )
            if gap < s.verify_min_gap_hours:
                continue

        new_goldens.append(
            GoldenRow(
                row_id=rid,
                query=cand["query"],
                canonical_url=payload["canonical"],
                equivalence_members=payload["members"],
                truth_token=cand["truth_token"],
                authoritative_timestamp=cand["authoritative_timestamp"],
                stratum=cand["stratum"],
                source=cand["source"],
                split="",  # assigned by bench_core.split
                verified_at=times[-2:] if not single_pass else [now],
                token_depth=payload["token_depth"],
                canonical_chars=payload["canonical_chars"],
                query_variants=cand.get("query_variants", {}),
                slices=assign_slices(cand),
            ).to_dict()
        )
        promoted += 1

    if new_goldens:
        write_jsonl(GOLDEN_FILE, new_goldens, append=True)
    print(f"[verify] {promoted} candidates promoted to golden")
    return promoted


def run_sync(single_pass: bool = False, limit: int | None = None) -> int:
    return asyncio.run(run(single_pass=single_pass, limit=limit))
