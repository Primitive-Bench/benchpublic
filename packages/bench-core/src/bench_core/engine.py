"""The run loop: dataset -> adapter -> scorer -> ItemResult.

Frozen interfaces for v0.1.0. The eval-* packages subclass Task and Scorer;
the CLI calls run_task. Adapter invocation is delegated to bench-adapters via a
duck-typed `adapter.invoke(item)` so bench-core has no hard dep on bench-adapters.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from bench_schemas import AdapterSpec, ItemResult, Primitive, ScorerOutput
from bench_schemas.models import GroundTruthTier


@runtime_checkable
class Adapter(Protocol):
    """Minimal adapter contract bench-core relies on (implemented in bench-adapters)."""

    spec: AdapterSpec

    def invoke(
        self, item: dict[str, Any]
    ) -> dict[str, Any]:  # returns {raw_output, latency_ms, cost_usd, ...}
        ...


class Scorer:
    """Subclass per primitive. Maps (item, adapter raw output) -> ScorerOutput."""

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        raise NotImplementedError


class Task:
    """A primitive eval task: a golden dataset + a scorer + slice tagging.

    Subclasses live in eval-* packages. They MUST be deterministic given a seed.
    """

    primitive: Primitive
    task_version: str
    dataset_version: str

    def items(self) -> Iterable[dict[str, Any]]:
        """Yield golden items. Each item carries at least: id, slices, ground_truth_tier."""
        raise NotImplementedError

    def scorer(self) -> Scorer:
        raise NotImplementedError


def run_task(task: Task, adapter: Adapter, run_id: str, seed: int) -> Iterable[ItemResult]:
    """Execute one (task, adapter) pass, yielding ItemResult records to stream as JSONL.

    Stub: the real engine adds ret/concurrency, cost capture, and error isolation.
    """
    scorer = task.scorer()
    for item in task.items():
        try:
            raw = adapter.invoke(item)
            out = scorer.score(item, raw)
            err = None
        except Exception as e:  # adapter/scorer failures isolate to the item, never crash the run
            raw, out, err = {}, ScorerOutput(correct=False, miss_reason="adapter_error"), str(e)
        tier = item.get("ground_truth_tier")
        yield ItemResult(
            run_id=run_id,
            adapter=adapter.spec.name,
            item_id=str(item["id"]),
            primitive=task.primitive,
            slices=item.get("slices", []),
            ground_truth_tier=GroundTruthTier(tier) if tier else None,
            output=out,
            raw_output=raw.get("raw_output"),
            latency_ms=raw.get("latency_ms"),
            cost_usd=raw.get("cost_usd"),
            error=err,
        )
