"""hit@k set membership + miss taxonomy + sentinel retrievability verdict.

Ported from arlenk2021/GoldenEvalsWebSearch `src/probe/scoring.py`. The liveness
gate itself lives in bench-core (`bench_core.verify.liveness_gate`) and is imported
where the probe needs it; here we keep the search-specific scoring instruments:

  * hit@k as PURE set membership after URL normalization against the row's
    equivalence class — no judge, no human.
  * the web_search MISS TAXONOMY (ranked_below_k / mirror_not_in_class / not_found).
  * the SENTINEL retrievability verdict (the only honest way to split a
    `not_found` miss into not_indexed vs ranked-deeper-than-k).
  * `find_promotions`, the mirror auto-promotion gate.

hit@k is delegated to `bench_stats.hit_at_k` for the canonical metric; a thin
local wrapper preserves the original `(returned_urls, members, ks)` signature used
by the probe so an unanticipated mirror promoted by vendor A counts for the batch.
"""

from __future__ import annotations

from bench_core.http import fetch
from bench_core.urls import EquivalenceClass


def hit_at_k(returned_urls: list[str], members: list[str], ks: tuple[int, ...]) -> dict[int, int]:
    """1 if any of the top-k returned URLs is in the equivalence class, else 0.

    Set membership against the row's equivalence class after URL normalization
    (normalization lives in bench_core.urls.EquivalenceClass). No judge, no human.
    """
    eq = EquivalenceClass(members[0], members[1:]) if members else None
    out: dict[int, int] = {}
    for k in ks:
        hit = 0
        if eq:
            for url in returned_urls[:k]:
                if eq.contains(url):
                    hit = 1
                    break
        out[k] = hit
    return out


def first_correct_rank(returned_urls: list[str], eq: EquivalenceClass) -> int:
    """1-based rank of the first returned URL in the class, or -1 if none."""
    for i, url in enumerate(returned_urls):
        if eq.contains(url):
            return i + 1
    return -1


# ---- web_search miss taxonomy (gap 3) ----------------------------------------
# A miss has distinct causes that must not be conflated (the §4 lesson):
#   ranked_below_k    a correct URL was returned, but below the cutoff k
#   mirror_not_in_class  a returned URL holds the truth token but wasn't yet a
#                     declared member -> auto-promoted; the row was being
#                     mis-scored as a miss (the single biggest correctness risk)
#   not_found         no returned URL (at any returned rank) is correct or holds
#                     the token. CANNOT be split into "not indexed" vs "ranked
#                     deeper than our result depth" WITHOUT the sentinel
#                     retrievability instrument (or a per-vendor direct URL lookup).
MISS_RANKED_BELOW_K = "ranked_below_k"
MISS_MIRROR_PROMOTED = "mirror_not_in_class"
MISS_NOT_FOUND = "not_found"

# ---- sentinel retrievability verdict -----------------------------------------
# Only sentinel rows can split a `not_found` miss into its two real causes,
# because the bench owns the page and minted a globally-unique truth token. On a
# descriptive miss we issue ONE extra "indexing probe" with the token_in_query
# variant; whether even that navigational query surfaces the canonical URL tells
# us whether the page is in the vendor index at all:
#   ranked_below_k  the page IS indexed (the token query found it) — the
#                   descriptive query merely failed to rank it in top-k.
#   not_indexed     even the unique-token query could not surface it — the vendor
#                   has not crawled/indexed the page yet (true index lag).
SENTINEL_RANKED_BELOW_K = "ranked_below_k"
SENTINEL_NOT_INDEXED = "not_indexed"


def classify_miss(
    returned: list[str], eq: EquivalenceClass, promoted: bool, ks: tuple[int, ...]
) -> str:
    """Decompose a miss into the taxonomy. `eq` already includes any promotions.

    `ks` is the pinned set-membership window; the cutoff is max(ks).
    """
    rank = first_correct_rank(returned, eq)  # eq already includes promotions
    if promoted and rank != -1 and rank > max(ks):
        return MISS_RANKED_BELOW_K
    if promoted:
        return MISS_MIRROR_PROMOTED  # promoted but still below k
    if rank != -1:
        return MISS_RANKED_BELOW_K
    return MISS_NOT_FOUND  # not_indexed-or-deeper (needs sentinel)


async def find_promotions(
    returned_urls: list[str], eq: EquivalenceClass, truth_token: str, depth: int
) -> list[tuple[str, str]]:
    """Returned URLs (top `depth`) NOT already in the class but whose page holds
    the truth token. These are unanticipated mirrors and must be auto-promoted
    (returns (returned_url, final_url) pairs). Same token-on-page gate as verify;
    cross-domain is allowed here because a search answer can legitimately live on
    a different registrable domain (cve.org vs nvd.nist.gov, govinfo vs FR)."""
    out: list[tuple[str, str]] = []
    for url in returned_urls[:depth]:
        if eq.contains(url):
            continue
        res = await fetch(url)
        if res.status == 200 and not res.soft_404 and truth_token in res.main_text:
            out.append((url, res.final_url))
    return out
