"""Token-survival scoring + miss classification for web_extraction.

Ported from arlenk2021/GoldenEvalsWebSearch (src/probe/extract/main.py: the
token_locate / classify_miss logic), re-expressed against the frozen
`bench_schemas.ScorerOutput` contract.

The model: hand a vendor the canonical URL of a golden row; the cell is a HIT iff
the row's `truth_token` survives extraction into the vendor's main content. The
token match is whitespace-normalized substring; Federal Register doc numbers,
CVE IDs, and SEC accession numbers are exact strings.

Miss decomposition is the methodology highlight. A miss is one of:

  * blocked       — the vendor hit an anti-bot / access-wall interstitial.
                    THE GAP WAS ANTI-BOT BLOCKING, NOT FRESHNESS: when Exa scores
                    33% token survival on the Federal Register, it is NOT because
                    the facts went stale — 98/100 of its misses carry a
                    "request access" / "just a moment" / Cloudflare interstitial.
                    That is a DIFFERENT capability gap (bot-blocking) than
                    extracting real content that happened to lack the token, so we
                    classify it separately and surface it on ScorerOutput.miss_reason.
  * truncated     — the token exists on the page but sits past where the vendor's
                    extraction was cut off (depth-conditional survival; see
                    token_locate). FR doc numbers live deep in the body and die
                    under truncation, while CVE IDs live at offset ~0 and survive.
  * token_absent  — real content was extracted, the token simply is not in it.
  * empty         — the vendor returned nothing usable.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from bench_schemas import ScorerOutput

_WS = re.compile(r"\s+")

# Anti-bot / access-wall interstitials. A miss because the vendor got one of
# these is a DIFFERENT capability gap (bot-blocking) than extracting real content
# that happened to lack the token — so we classify it separately. This is the
# instrumentation behind "the gap was anti-bot blocking, not freshness": these
# signatures are exactly what 98/100 of Exa's Federal Register misses match.
_BLOCK_SIGNATURES = (
    "request access", "aggressive automated scraping", "just a moment",
    "attention required", "verify you are human", "are you a robot",
    "captcha", "access denied", "enable javascript", "cloudflare",
    "unusual traffic", "403 forbidden",
)


def classify_miss(token: str, text: str, *, token_depth: int = -1) -> str:
    """Decompose WHY the truth token did not survive extraction.

    Returns one of: 'empty', 'blocked', 'truncated', 'token_absent'.

    The order is deliberate. An anti-bot interstitial is checked FIRST: when a
    vendor is firewalled, the "content" it returns is the block page, and the
    truth token's absence from a block page tells us nothing about extraction
    quality — only about bot-blocking. Conflating the two is the exact mistake the
    miss decomposition exists to prevent (the gap was anti-bot blocking, not
    freshness, and not extraction skill).

    `token_depth` is the offset of the token in the CANONICAL (source-of-truth)
    main content. If the token lives deep in the body on the gold page but the
    vendor returned a non-trivial prefix that simply ends before that depth, the
    miss is `truncated` rather than `token_absent` — depth-conditional survival
    defuses the confound that snippet-only extractors look good on title-token
    sources but die on deep-body tokens (FR doc numbers)."""
    if not text or not text.strip():
        return "empty"
    low = text.lower()
    if any(sig in low for sig in _BLOCK_SIGNATURES):
        return "blocked"
    # Truncation: the gold page carries the token deep in the body, and the
    # vendor returned real content whose length never reaches that depth.
    norm_len = len(_WS.sub(" ", text))
    if token_depth >= 0 and norm_len > 0 and norm_len < token_depth:
        return "truncated"
    return "token_absent"


def token_locate(token: str, text: str) -> tuple[int, int]:
    """Return (offset, total_len) of the token in whitespace-normalized text.

    offset is the char index where the token first appears, or -1 if absent.
    Recording the offset lets us separate "token survived" from "token survived
    only because it sat in the title/first snippet" — CVE IDs live at offset ~0
    (title/URL) and survive any truncation, while FR doc numbers live deep in the
    body and die under truncation. Depth-conditional survival defuses that confound.
    """
    if not text:
        return -1, 0
    norm = _WS.sub(" ", text)
    idx = norm.find(token)
    if idx < 0:
        # tolerate internal-whitespace differences; offset then unknown -> 0-len marker
        if _WS.sub("", token) in _WS.sub("", text):
            return 0, len(norm)
        return -1, len(norm)
    return idx, len(norm)


def token_survives(token: str, text: str) -> bool:
    return token_locate(token, text)[0] >= 0


def score_extraction(
    item: dict[str, Any],
    main_text: str,
    *,
    error: Optional[str] = None,
) -> ScorerOutput:
    """Token-survival score for one (vendor, golden row) cell -> ScorerOutput.

    `item` carries at least `truth_token`; `token_depth` (offset in canonical
    main content) is used to distinguish truncated from token_absent when present.
    `main_text` is the vendor's extracted main content. `error` short-circuits to
    a transport failure (no token charged).

    The returned ScorerOutput sets:
      * correct        — did the truth token survive extraction (the HIT)
      * score          — 1.0/0.0 mirror of correct, for continuous aggregation
      * miss_reason    — the decomposed failure category (None on a hit)
      * metrics        — token_offset / chars / norm_chars for depth-conditional
                         analysis (the title-zone vs deep-body confound)
    """
    token = str(item["truth_token"])
    token_depth = int(item.get("token_depth", -1))

    if error:
        # Transport failure: nothing scoreable. Not charged a token miss.
        return ScorerOutput(
            correct=None,
            score=None,
            miss_reason="fetch_failed",
            rationale=str(error)[:200],
        )

    offset, norm_len = token_locate(token, main_text)
    hit = offset >= 0
    miss_reason = None if hit else classify_miss(token, main_text, token_depth=token_depth)

    return ScorerOutput(
        correct=hit,
        score=1.0 if hit else 0.0,
        miss_reason=miss_reason,
        metrics={
            "token_offset": float(offset),   # -1 absent; ~0 = title/snippet zone
            "chars": float(len(main_text or "")),
            "norm_chars": float(norm_len),
        },
    )
