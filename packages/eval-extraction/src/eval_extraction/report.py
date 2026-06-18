"""Extraction leaderboard: per-vendor token-survival rate with Wilson intervals.

Ported from arlenk2021/GoldenEvalsWebSearch (src/probe/extract/report.py),
adapted to consume `bench_schemas.ItemResult` records and the bench_stats Wilson
interval (`bench_stats.wilson` -> StatTest with statistic/ci_low/ci_high).

This is the report that tells the headline story. Token survival alone hides the
mechanism, so the board also decomposes every miss into blocked / truncated /
token_absent: a vendor at 33% survival because it is being firewalled
(`blocked`) is a fundamentally different result than one at 33% because the facts
went stale or it extracts poorly (`token_absent`). The miss breakdown is what
lets a reader see that the gap was anti-bot blocking, not freshness.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from bench_schemas import ItemResult

from bench_stats import wilson


def run(items: Iterable[ItemResult]) -> list[dict]:
    """Build the per-vendor token-survival board from scored ItemResults.

    Every vendor that was attempted appears, even if it ONLY errored (nothing
    silently dropped). The miss breakdown (blocked/truncated/token_absent/empty)
    accompanies each row so the anti-bot-blocking mechanism is legible.
    """
    by_vendor: dict[str, list[int]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    miss_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in items:
        if r.error:
            errors[r.adapter] += 1
            continue
        hit = r.output.correct
        if hit is None:
            # transport-level non-attempt (e.g. fetch_failed): count as availability
            errors[r.adapter] += 1
            continue
        by_vendor[r.adapter].append(int(hit))
        if not hit and r.output.miss_reason:
            miss_counts[r.adapter][r.output.miss_reason] += 1

    # Every vendor that was attempted appears, even if it ONLY errored.
    board: list[dict] = []
    for vendor in sorted(set(by_vendor) | set(errors)):
        hits = by_vendor.get(vendor, [])
        ci = wilson(sum(hits), len(hits)) if hits else None
        misses = dict(miss_counts.get(vendor, {}))
        board.append(
            {
                "vendor": vendor,
                "n": len(hits),
                "token_survival": round(ci.statistic, 4) if ci else None,
                "wilson_low": round(ci.ci_low, 4) if ci else None,
                "wilson_high": round(ci.ci_high, 4) if ci else None,
                "errors": errors.get(vendor, 0),
                # Miss decomposition — the anti-bot-blocking story. `blocked` here is
                # the count of misses that hit an access-wall, NOT genuine misses.
                "miss_breakdown": misses,
                "blocked": misses.get("blocked", 0),
                "truncated": misses.get("truncated", 0),
                "token_absent": misses.get("token_absent", 0),
            }
        )
    board.sort(key=lambda x: (x["token_survival"] is None, -(x["token_survival"] or 0)))

    print("[extract-report] token-survival rate by vendor:")
    for row in board:
        if row["token_survival"] is None:
            print(f"  {row['vendor']:>13}  --no scored rows--  (errors={row['errors']})")
        else:
            blk = row["blocked"]
            blk_note = f", blocked={blk}" if blk else ""
            print(
                f"  {row['vendor']:>13}  {row['token_survival']:.3f}  "
                f"[{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]  "
                f"(n={row['n']}, errors={row['errors']}{blk_note})"
            )
    return board
