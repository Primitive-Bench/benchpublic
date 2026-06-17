"""eval-extraction — Primitive Bench vertical (web_extraction).

URL -> clean main content, scored by TOKEN SURVIVAL: a cell is a hit iff the
golden row's truth_token survives the vendor's extraction. Every miss is
decomposed into blocked / truncated / token_absent so an anti-bot wall is never
confused with a genuine extraction miss (see scoring.py).

Implements bench_core.Task + bench_core.Scorer for this primitive. The public
golden DEV split lives in golden-sets-public/extraction/; the held-out test split
lives only behind the private eval server. Slice definitions: slices.yaml.

Provenance: ported from arlenk2021/GoldenEvalsWebSearch (extract probe +
token-survival report). See README.md.
"""

from eval_extraction.task import Scorer, Task, load_slices
from eval_extraction.scoring import (
    classify_miss,
    score_extraction,
    token_locate,
    token_survives,
)
from eval_extraction import probe, report

__all__ = [
    "Task",
    "Scorer",
    "load_slices",
    "score_extraction",
    "classify_miss",
    "token_locate",
    "token_survives",
    "probe",
    "report",
]
