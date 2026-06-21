"""eval-retrieval — Primitive Bench vertical (first-stage retrieval over BEIR sets).

Hand a bi-encoder a query + a fixed per-query candidate pool (qrels positives + BM25
hard negatives), get back the ranking, and score it with the BEIR/MTEB-standard IR
metrics (nDCG@10 / recall@10 / MAP / MRR). The leaderboard separability surface is
**success@10** — a paired-binary outcome the McNemar/Wilson gate consumes — so
per-slice winners (or TIE bands), never one global ranking.

Implements `bench_core.Task` + `bench_core.Scorer`. The public DEV split lives in
golden-sets-public/retrieval/ (a small committed example + a local BEIR generator);
the held-out test split lives only behind the private eval server. Slices:
slices.yaml (domain / relevant_set).
"""

from eval_retrieval.task import Task
from eval_retrieval.scoring import score_retrieval
from eval_retrieval.runner import run_sync

__all__ = ["Task", "score_retrieval", "run_sync"]
