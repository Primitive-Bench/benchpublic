"""Deterministic public/holdout split via HMAC-SHA256 (D-07 split-integrity).

split(row_id) = "public" if HMAC(salt, row_id) maps below the public fraction,
else "holdout". Applied independently per stratum so each stratum gets the same
70/30 ratio. The salt lives only in the environment (GitHub Secret / local
.env) and is never written to any artifact, so holdout membership is
uncomputable from the published data.
"""

from __future__ import annotations

import hashlib
import hmac

from bench_core.config import get_settings
from bench_core.domain import Split
from bench_core.storage import GOLDEN, SPLITS, read_jsonl, write_jsonl

GOLDEN_FILE = GOLDEN / "golden.jsonl"
SPLIT_FILE = SPLITS / "splits.jsonl"


def split_score(row_id: str, salt: str) -> float:
    """Uniform in [0, 1), deterministic given (salt, row_id)."""
    digest = hmac.new(salt.encode(), row_id.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(row_id: str, salt: str, public_fraction: float) -> str:
    """Raw-threshold assignment. Stable under row insertion but yields binomial
    drift in the holdout fraction (the bug at small n). Kept for reference/tests;
    hmac_split() uses the exact-quota allocator below."""
    return Split.PUBLIC if split_score(row_id, salt) < public_fraction else Split.HOLDOUT


def allocate_stratified(rows: list[dict], salt: str, public_fraction: float) -> list[dict]:
    """Exact-quota per-stratum split.

    Within each stratum, rank rows by HMAC(salt, row_id) and take exactly
    round((1 - public_fraction) * n) as holdout. This gives the spec's exact
    70/30 per stratum (no binomial drift) while staying deterministic and
    salt-dependent — holdout membership is still uncomputable without the salt.
    """
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    out: list[dict] = []
    for stratum, srows in by_stratum.items():
        ranked = sorted(srows, key=lambda r: split_score(r["row_id"], salt))
        n_holdout = round((1.0 - public_fraction) * len(ranked))
        # top n_holdout by score -> holdout; deterministic, exact count
        holdout_ids = (
            {r["row_id"] for r in ranked[len(ranked) - n_holdout :]} if n_holdout else set()
        )
        for r in srows:
            split = Split.HOLDOUT if r["row_id"] in holdout_ids else Split.PUBLIC
            out.append({**r, "split": split})
    return out


# Public contract name (INTERPACKAGE): exact-quota 70/30 per-stratum HMAC split.
hmac_split = allocate_stratified


def run() -> dict[str, int]:
    s = get_settings()
    salt = s.require_salt()
    rows = list(read_jsonl(GOLDEN_FILE))
    out = allocate_stratified(rows, salt, s.split_public_fraction)
    counts: dict[str, int] = {}
    for r in out:
        counts[f"{r['stratum']}/{r['split']}"] = counts.get(f"{r['stratum']}/{r['split']}", 0) + 1
    write_jsonl(SPLIT_FILE, out)
    print(f"[split] {len(out)} rows assigned (exact quota): {counts}")
    return counts


if __name__ == "__main__":
    run()
