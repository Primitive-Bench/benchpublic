"""Core domain record types + canonical row_id derivation.

A golden eval row is a fact frozen before any vendor is queried:
  query + equivalence-class of golden URLs + truth_token + authoritative_timestamp.

These are the rich bench-core domain types — richer than the frozen
`bench_schemas` contract models. They live here, and map to
`bench_schemas.ItemResult` / `bench_schemas.GroundTruthTier` at output
boundaries (see `stratum_to_tier`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from bench_schemas.models import GroundTruthTier


class Stratum(StrEnum):
    NAVIGATIONAL = "navigational"  # verified-external truth
    LONG_TAIL = "long_tail"  # verified-external truth
    FRESH = "fresh"  # authoritative-registry truth (delta pump)
    SENTINEL = "sentinel"  # truth by construction (bench-published)
    SOS = "sos"  # deferred in v1 (hand-curated, no pump)


class Split(StrEnum):
    PUBLIC = "public"
    HOLDOUT = "holdout"


# Mapping from the rich bench-core Stratum to the frozen bench_schemas
# GroundTruthTier, applied at output boundaries (e.g. emitting ItemResult):
#   navigational / long_tail -> VERIFIED_EXTERNAL
#   fresh                     -> AUTHORITATIVE_REGISTRY
#   sentinel                  -> SENTINEL_PLANTED
_STRATUM_TO_TIER: dict[str, GroundTruthTier] = {
    Stratum.NAVIGATIONAL: GroundTruthTier.VERIFIED_EXTERNAL,
    Stratum.LONG_TAIL: GroundTruthTier.VERIFIED_EXTERNAL,
    Stratum.FRESH: GroundTruthTier.AUTHORITATIVE_REGISTRY,
    Stratum.SENTINEL: GroundTruthTier.SENTINEL_PLANTED,
}


def stratum_to_tier(stratum: str) -> GroundTruthTier | None:
    """Map a bench-core Stratum onto the frozen bench_schemas GroundTruthTier.

    Returns None for strata with no contract tier (e.g. the deferred `sos`).
    """
    return _STRATUM_TO_TIER.get(stratum)


def derive_row_id(stratum: str, golden_url: str, truth_token: str) -> str:
    """Stable id independent of split secret. Used as the HMAC message.

    Deterministic from the row's identity so the same fact always lands in the
    same split bucket, and so re-running the pump never reshuffles assignments.
    """
    h = hashlib.sha256(f"{stratum}\x1f{golden_url}\x1f{truth_token}".encode())
    return h.hexdigest()[:24]


@dataclass
class Candidate:
    """Pump output: a proposed row, not yet verified."""

    query: str
    golden_url: str
    truth_token: str
    authoritative_timestamp: str  # ISO-8601 from the source of truth
    stratum: str
    source: str  # e.g. "sec_edgar", "nvd_cve"
    row_id: str = field(default="")
    # query variants by form for the web_search primitive. The default `query`
    # is the descriptive (honest-discriminator) form; token_in_query is the easy
    # navigational form that title-zone-style saturates.
    query_variants: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.row_id:
            self.row_id = derive_row_id(self.stratum, self.golden_url, self.truth_token)

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "query": self.query,
            "golden_url": self.golden_url,
            "truth_token": self.truth_token,
            "authoritative_timestamp": self.authoritative_timestamp,
            "stratum": self.stratum,
            "source": self.source,
            "query_variants": self.query_variants,
        }


@dataclass
class GoldenRow:
    """A verified, split-assigned row ready to be probed."""

    row_id: str
    query: str
    canonical_url: str
    equivalence_members: list[str]
    truth_token: str
    authoritative_timestamp: str
    stratum: str
    source: str
    split: str
    verified_at: list[str]  # the two (or more) verification fetch timestamps
    token_depth: int = -1  # token offset in the CANONICAL main content
    canonical_chars: int = 0  # length of canonical main content
    query_variants: dict[str, str] = field(default_factory=dict)
    slices: list[str] = field(default_factory=list)  # cross-cutting intent tags

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "query": self.query,
            "canonical_url": self.canonical_url,
            "equivalence_members": self.equivalence_members,
            "truth_token": self.truth_token,
            "authoritative_timestamp": self.authoritative_timestamp,
            "stratum": self.stratum,
            "source": self.source,
            "split": self.split,
            "verified_at": self.verified_at,
            "token_depth": self.token_depth,
            "canonical_chars": self.canonical_chars,
            "query_variants": self.query_variants,
            "slices": self.slices,
        }
