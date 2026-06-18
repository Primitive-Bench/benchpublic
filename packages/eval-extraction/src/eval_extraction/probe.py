"""web_extraction probe.

Ported from arlenk2021/GoldenEvalsWebSearch (src/probe/extract/main.py).

For each golden row, hand the canonical URL to each extraction vendor and score
hit = (truth token survives extraction). Same liveness contract as web_search:
a row whose own gold page no longer carries the token in main content is excluded
batch-wide, never charged to a vendor.

Adapted to the public monorepo: vendors are bench_adapters extraction adapters
(firecrawl, jina, exa_live, tavily_extract, apify), each invoked through the
uniform `Adapter.invoke(item) -> dict` interface (returning `main_text`). Each
scored cell is emitted as a `bench_schemas.ItemResult`. Token match is
whitespace-normalized substring; FR doc numbers / CVE IDs / accession numbers are
exact strings (handled in scoring.token_locate).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from bench_schemas import AdapterSpec, ItemResult, ScorerOutput
from bench_schemas.models import GroundTruthTier, Primitive

from bench_adapters import get, registry
from bench_core.domain import stratum_to_tier
from bench_core.verify import liveness_gate

from eval_extraction.scoring import score_extraction

# Default extraction adapters registered in bench_adapters (see INTERPACKAGE.md).
EXTRACTION_VENDORS = ("firecrawl", "jina", "exa_live", "tavily_extract", "apify")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tier_for(row: dict) -> Optional[GroundTruthTier]:
    """Map the row's stratum onto the frozen GroundTruthTier (None if unmapped)."""
    return stratum_to_tier(row.get("stratum", ""))


def _adapter(name: str) -> Any:
    """Instantiate one extraction adapter by registry name.

    Adapters take an AdapterSpec; we synthesize a minimal spec from the name so
    the probe can run any registered extraction vendor without a hand-authored
    config. The harness may pass a richer AdapterSpec instead.
    """
    cls = get(name)
    spec = AdapterSpec(name=name, primitive=Primitive.EXTRACTION, vendor=name, version="live")
    return cls(spec)


def _main_text(raw: dict[str, Any]) -> str:
    """Pull the extracted main content out of a bench_adapters result dict.

    Extraction adapters return `main_text` (INTERPACKAGE.md); we tolerate a few
    aliases so an adapter that names it differently is not silently scored empty.
    """
    return str(raw.get("main_text") or raw.get("content") or raw.get("raw_output") or "")


async def _extract_row(
    row: dict, run_id: str, vendors: list[tuple[str, Any]], reps: int
) -> list[ItemResult]:
    """Score one golden row against every vendor (reps times) -> ItemResult list.

    `vendors` is a list of (name, adapter) pairs. A vendor whose key is unset is
    skipped cleanly (no result emitted, never charged a miss it had no chance at);
    any other exception is recorded as a failed ItemResult with `error` set.
    """
    out: list[ItemResult] = []
    item = {"truth_token": row["truth_token"], "token_depth": row.get("token_depth", -1)}
    slices = list(row.get("slices", []))
    tier = _tier_for(row)
    for name, adapter in vendors:
        for rep in range(reps):
            invoke_item = {"url": row["canonical_url"], **item}
            try:
                raw = adapter.invoke(invoke_item)
            except Exception as exc:  # missing key / transport — isolate to the cell
                out.append(
                    ItemResult(
                        run_id=run_id,
                        adapter=name,
                        item_id=row["row_id"],
                        primitive=Primitive.EXTRACTION,
                        slices=slices,
                        ground_truth_tier=tier,
                        output=ScorerOutput(correct=None, miss_reason="fetch_failed"),
                        error=repr(exc)[:200],
                    )
                )
                continue
            text = _main_text(raw)
            score = score_extraction(item, text)
            out.append(
                ItemResult(
                    run_id=run_id,
                    adapter=name,
                    item_id=row["row_id"],
                    primitive=Primitive.EXTRACTION,
                    slices=slices,
                    ground_truth_tier=tier,
                    output=score,
                    raw_output=raw.get("raw_output"),
                    latency_ms=raw.get("latency_ms"),
                    cost_usd=raw.get("cost_usd"),
                )
            )
    return out


async def run(
    rows: Iterable[dict],
    run_id: str,
    *,
    vendor: str = "all",
    limit: int | None = None,
    reps: int | None = None,
    skip_liveness: bool = False,
) -> list[ItemResult]:
    """Run the extraction probe over `rows`, returning scored ItemResults.

    Liveness contract: each row is re-checked with `bench_core.verify.liveness_gate`
    before any vendor is scored against it. A row whose own gold page no longer
    carries the truth token in main content is excluded batch-wide — never charged
    to a vendor (the same "truth checked at use" gate as web_search). Pass
    `skip_liveness=True` for offline/replay runs where rows are pre-verified.

    `vendor` accepts "all", a single name, or a comma-separated subset. Extraction
    is deterministic-ish, so `reps` defaults to 1.
    """
    rows = list(rows)
    if limit:
        rows = rows[:limit]
    # accept "all", a single name, or a comma-separated subset
    names = list(EXTRACTION_VENDORS) if vendor == "all" else [v.strip() for v in vendor.split(",")]
    vendors = [(n, _adapter(n)) for n in names if n in registry]
    reps = reps if reps is not None else 1  # extraction is deterministic-ish; default 1

    live: list[dict] = []
    if skip_liveness:
        live = rows
    else:
        for r in rows:
            lv = await liveness_gate(
                {
                    "golden_url": r["canonical_url"],
                    "truth_token": r["truth_token"],
                    "source": r.get("source"),
                }
            )
            if lv.live:
                live.append(r)
            # else: dropped batch-wide, never charged to a vendor (logged upstream)

    records: list[ItemResult] = []
    for r in live:
        records.extend(await _extract_row(r, run_id, vendors, reps))
    scored = [x for x in records if x.error is None]
    print(f"[extract] {len(live)}/{len(rows)} rows live; {len(scored)} extraction results")
    return records


def run_sync(
    rows: Iterable[dict],
    run_id: str,
    *,
    vendor: str = "all",
    limit: int | None = None,
    reps: int | None = None,
    skip_liveness: bool = False,
) -> list[ItemResult]:
    return asyncio.run(
        run(
            rows,
            run_id,
            vendor=vendor,
            limit=limit,
            reps=reps,
            skip_liveness=skip_liveness,
        )
    )
