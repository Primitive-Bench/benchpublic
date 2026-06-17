"""extraction Task + Scorer (web_extraction vertical).

Ported from arlenk2021/GoldenEvalsWebSearch (extract probe). The Scorer wraps the
token-survival model in `scoring.py`; the Task yields golden rows (canonical URL +
truth_token) and tags them with the extraction slices from slices.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from bench_core import Scorer as _Scorer, Task as _Task
from bench_schemas import ScorerOutput
from bench_schemas.models import Primitive

from eval_extraction.scoring import score_extraction

_SLICES_FILE = Path(__file__).resolve().parents[2] / "slices.yaml"


class Scorer(_Scorer):
    """Token-survival scorer: hit iff the row's truth_token survives extraction.

    `raw` is the bench_adapters extraction result dict (`main_text` = the vendor's
    extracted main content). Misses are decomposed (blocked / truncated /
    token_absent) onto ScorerOutput.miss_reason — see scoring.classify_miss.
    """

    def score(self, item: dict[str, Any], raw: dict[str, Any]) -> ScorerOutput:
        main_text = str(
            raw.get("main_text") or raw.get("content") or raw.get("raw_output") or ""
        )
        return score_extraction(item, main_text, error=raw.get("error"))


class Task(_Task):
    primitive = Primitive.EXTRACTION
    task_version = "extraction@1"
    dataset_version = "extraction-2026.06"

    def __init__(self, rows: Iterable[dict[str, Any]] | None = None) -> None:
        """`rows` are golden rows (canonical_url + truth_token + stratum/source/…).

        When omitted the Task carries no rows (the public DEV split is loaded by
        the harness from golden-sets-public/extraction/). Each yielded item is
        normalized to the shape the Scorer expects: id, truth_token, token_depth,
        url, slices, ground_truth_tier.
        """
        self._rows = list(rows or [])

    def items(self) -> Iterable[dict[str, Any]]:
        for r in self._rows:
            yield {
                "id": r["row_id"],
                "truth_token": r["truth_token"],
                "token_depth": r.get("token_depth", -1),
                "url": r["canonical_url"],
                "slices": list(r.get("slices", [])),
                "ground_truth_tier": _tier(r),
            }

    def scorer(self) -> _Scorer:
        return Scorer()


def _tier(row: dict[str, Any]) -> str | None:
    """Map the row's stratum onto the frozen GroundTruthTier value (or None)."""
    from bench_core.domain import stratum_to_tier

    t = stratum_to_tier(row.get("stratum", ""))
    return t.value if t else None


def load_slices() -> list[dict[str, Any]]:
    """Read the extraction slice definitions from slices.yaml."""
    data = yaml.safe_load(_SLICES_FILE.read_text())
    return list(data.get("slices", []))
