"""websearch Task + Scorer — wires golden rows -> probe -> score.

Subclasses bench_core.Task / bench_core.Scorer for the web_search primitive.
The public golden DEV split lives in golden-sets-public/websearch/; the held-out
test split lives only behind the private eval server.

The Scorer maps one vendor's returned URLs (the bench-adapters search result
dict, `{returned_urls, ...}`) for a golden item to a `bench_schemas.ScorerOutput`:
hit@{1,5} set membership against the row's equivalence class, the miss taxonomy
on `output.miss_reason`, and the equivalence-class id. The full async probe
loop (liveness gate, mirror auto-promotion, sentinel verdict) lives in
`eval_websearch.probe`; this Scorer is the synchronous per-item view bench-core's
`run_task` drives.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from bench_core import Scorer as _Scorer, Task as _Task
from bench_core.domain import stratum_to_tier
from bench_core.urls import EquivalenceClass
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_websearch.probe import KS
from eval_websearch.queries import DEFAULT_FORM
from eval_websearch.scoring import classify_miss, first_correct_rank, hit_at_k

# Public DEV split location (overridable). The held-out split never ships here.
_DEFAULT_GOLDEN = "golden-sets-public/websearch/dev.jsonl"


class Scorer(_Scorer):
    """Score one vendor's search result for one golden item (set membership).

    `item` is a golden row dict (carrying `equivalence_members`, `truth_token`,
    `slices`, `stratum`). `raw` is the bench-adapters search result dict; its
    `returned_urls` are scored as hit@k against the row's equivalence class.

    NOTE: mirror auto-promotion and the sentinel verdict require async re-fetches
    and are applied by `eval_websearch.probe`. This synchronous scorer reflects
    set membership against the row's PRE-DECLARED equivalence class only, so its
    `miss_reason` distinguishes `ranked_below_k` from `not_found` but never
    `mirror_not_in_class` (no promotion is performed here).
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        members = item["equivalence_members"]
        eq = EquivalenceClass(members[0], members[1:])
        urls = list(raw.get("returned_urls", []))
        h = hit_at_k(urls, eq.members, KS)
        correct = bool(h[max(KS)])
        miss_reason = None if correct else classify_miss(urls, eq, promoted=False, ks=KS)
        metrics = {f"hit@{k}": float(h[k]) for k in KS}
        metrics["first_rank"] = float(first_correct_rank(urls, eq))
        metrics["n_results"] = float(len(urls))
        if raw.get("latency_ms") is not None:
            metrics["latency_ms"] = float(raw["latency_ms"])
        return ScorerOutput(
            correct=correct,
            score=float(h[max(KS)]),
            metrics=metrics,
            miss_reason=miss_reason,
            equivalence_class=eq.canonical,
        )


class Task(_Task):
    primitive = Primitive.WEBSEARCH
    task_version = "websearch@1"
    dataset_version = "websearch-2026.06"

    def __init__(
        self, golden_path: str | os.PathLike[str] | None = None, form: str = DEFAULT_FORM
    ) -> None:
        self.golden_path = Path(golden_path or os.environ.get("WEBSEARCH_GOLDEN", _DEFAULT_GOLDEN))
        self.form = form

    def items(self) -> Iterable[dict[str, Any]]:
        """Yield golden rows from the DEV split JSONL.

        Each yielded item carries `id` (the bench-core run loop key), the chosen
        query form on `query`, `slices`, and `ground_truth_tier` (mapped from the
        row Stratum), plus the scoring payload (`equivalence_members`,
        `truth_token`).
        """
        if not self.golden_path.exists():
            return
        with self.golden_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                variants = row.get("query_variants") or {}
                query = variants.get(self.form) or row.get("query")
                if query is None:
                    continue
                tier = stratum_to_tier(row.get("stratum", ""))
                yield {
                    **row,
                    "id": row["row_id"],
                    "query": query,
                    "slices": row.get("slices", []),
                    "ground_truth_tier": tier.value if tier else None,
                }

    def scorer(self) -> _Scorer:
        return Scorer()
