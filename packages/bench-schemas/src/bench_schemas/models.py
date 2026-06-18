"""Frozen contract models for Primitive Bench (v0.1.0).

Design rules (enforced by review, see DECISIONS.md D-03):
  * Additive-only within a MINOR version. New OPTIONAL fields are fine; renaming
    or removing a field, or tightening a type, is a MAJOR bump.
  * Every package imports types ONLY from here and writes ONLY files it owns.
    No shared mutable state — this is what keeps the agent lanes collision-free.
  * Results are emitted as JSONL: one ItemResult per line during a run; one
    RunManifest per run; SliceResult objects are derived (by bench-stats) and
    re-emitted, never hand-authored.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Bumped on every change to this file. The ingest pipeline keys on it.
SCHEMA_VERSION = "0.1.0"


class Primitive(str, Enum):
    """The infrastructure primitives Primitive Bench certifies, one vertical each."""

    OCR = "ocr"
    WEBSEARCH = "websearch"
    VECTORDB = "vectordb"
    RERANKER = "reranker"
    RETRIEVAL = "retrieval"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    CRAWL = "crawl"
    MEMORY = "memory"


class GroundTruthTier(str, Enum):
    """Three-tier ground-truth model from the WebSearch methodology."""

    VERIFIED_EXTERNAL = "verified_external"  # human-verified against an external source
    AUTHORITATIVE_REGISTRY = "authoritative_registry"  # canonical registry of record
    SENTINEL_PLANTED = "sentinel_planted"  # known item planted to detect drift/contamination


# ---------------------------------------------------------------------------
# AdapterSpec — declares one provider/primitive adapter (lm-eval registry pattern)
# ---------------------------------------------------------------------------
class AdapterSpec(BaseModel):
    """Identifies a system-under-test and how to invoke it deterministically."""

    name: str = Field(..., description="Stable adapter id, e.g. 'claude-sonnet-ocr', 'qdrant'")
    primitive: Primitive
    vendor: str = Field(..., description="Vendor/org, e.g. 'anthropic', 'qdrant'")
    version: str = Field(..., description="Model/engine version pin, e.g. 'claude-sonnet-4-6'")
    is_sentinel: bool = Field(
        default=False,
        description="Regression sentinel (e.g. Tesseract). Expected to be stable, not to win.",
    )
    # Free-form, adapter-specific knobs (temperature, top_k, ef_search, ...).
    params: dict[str, Any] = Field(default_factory=dict)
    # EULA flag: some vendors (DeWitt clause) forbid publishing benchmark results.
    publish_restricted: bool = Field(
        default=False,
        description="True if the vendor EULA restricts publishing results (BenchANT: 4/13 vector DBs).",
    )


# ---------------------------------------------------------------------------
# ScorerOutput — what a scorer emits for a single item
# ---------------------------------------------------------------------------
class ScorerOutput(BaseModel):
    """Per-item score. `correct` drives proportion stats (Wilson/McNemar);
    `metrics` carries continuous metrics (nDCG, latency_ms, ...) for bootstrap CIs."""

    correct: Optional[bool] = Field(
        default=None, description="Binary pass/fail for proportion-based slices (hit@k, accuracy)."
    )
    score: Optional[float] = Field(
        default=None, description="Primary continuous score if applicable."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Named continuous metrics, e.g. {'ndcg@10': 0.82, 'latency_ms': 41.0}.",
    )
    # Miss decomposition: WHY an item failed (the WebSearch methodology requires this).
    miss_reason: Optional[str] = Field(
        default=None, description="Decomposed failure category, e.g. 'not_retrieved', 'wrong_rank'."
    )
    # Equivalence-class id: items the scorer treats as interchangeably-correct.
    equivalence_class: Optional[str] = Field(default=None)
    rationale: Optional[str] = Field(default=None, description="Optional grader explanation.")


# ---------------------------------------------------------------------------
# ItemResult — the atomic JSONL record written during a run
# ---------------------------------------------------------------------------
class ItemResult(BaseModel):
    """One (adapter, item) evaluation. Streamed to <run>/items.jsonl."""

    run_id: str
    adapter: str = Field(..., description="AdapterSpec.name")
    item_id: str = Field(..., description="Stable id of the golden item")
    primitive: Primitive
    # Slice/constraint tags this item participates in (e.g. ['doc_type:invoice', 'lang:en']).
    slices: list[str] = Field(default_factory=list)
    ground_truth_tier: Optional[GroundTruthTier] = None
    output: ScorerOutput
    raw_output: Optional[str] = Field(default=None, description="Raw system output, for audit.")
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = Field(default=None, description="Set if the adapter call failed.")


# ---------------------------------------------------------------------------
# StatTest — a statistical result attached to a slice
# ---------------------------------------------------------------------------
class StatTest(BaseModel):
    """Output of a bench-stats test, carried on SliceResult for separability badges."""

    method: Literal["mcnemar", "wilson", "bootstrap", "bradley_terry"]
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    seed: Optional[int] = Field(
        default=None, description="Fixed seed for bootstrap reproducibility."
    )


# ---------------------------------------------------------------------------
# SliceResult — aggregated result for one slice (derived, never hand-authored)
# ---------------------------------------------------------------------------
class SliceResult(BaseModel):
    """Per-slice aggregate for one adapter. The unit the leaderboard ranks.

    `separable` is the trust gate: if False (overlapping CIs / high McNemar p at
    this n), the leaderboard MUST NOT publish a single winner for the slice.
    """

    run_id: str
    primitive: Primitive
    slice: str = Field(..., description="Slice/constraint key, e.g. 'doc_type:invoice'")
    adapter: str
    n: int = Field(..., description="Number of items in this slice for this adapter.")
    point_estimate: float = Field(..., description="Primary metric value (accuracy, nDCG, ...).")
    metric_name: str = Field(default="accuracy")
    ci: Optional[StatTest] = Field(
        default=None, description="Wilson/bootstrap CI for the estimate."
    )
    separable: Optional[bool] = Field(
        default=None,
        description="Whether this adapter is statistically separable from the runner-up on this slice.",
    )
    rank: Optional[int] = None


# ---------------------------------------------------------------------------
# RunManifest — describes one reproducible run (written once per run)
# ---------------------------------------------------------------------------
class RunManifest(BaseModel):
    """The reproducibility record. Written to <run>/manifest.json.

    Anyone with the public dev split + this manifest can reproduce a public run.
    Held-out runs reference the manifest but score on the private eval server.
    """

    run_id: str
    schema_version: str = Field(default=SCHEMA_VERSION)
    primitive: Primitive
    created_at: datetime
    seed: int = Field(..., description="Master deterministic seed for the run.")
    adapters: list[AdapterSpec]
    # Versioning for reproducibility (lm-eval task-versioning pattern).
    task_version: str = Field(..., description="Version of the eval task/scorer definition.")
    dataset_version: str = Field(..., description="Pinned golden-set version (e.g. 'ocr-2026.06').")
    split: Literal["public_dev", "heldout_test"] = "public_dev"
    # HMAC commitment over the split membership — split-integrity differentiator (D-07).
    split_hmac: Optional[str] = Field(
        default=None, description="HMAC commitment of split membership."
    )
    # Canary marker embedded in golden files (BIG-bench convention) for contamination detection.
    canary_guid: Optional[str] = None
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Captured env: package versions, docker digests, hardware.",
    )
    notes: Optional[str] = None
