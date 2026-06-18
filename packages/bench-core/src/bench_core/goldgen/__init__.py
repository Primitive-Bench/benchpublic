"""Registry-delta pump orchestrator (goldgen).

Polls one or all authoritative registries (SEC EDGAR, Federal Register, NVD CVE,
GitHub Releases), de-dupes by row_id against rows already seen, and appends
golden Candidate rows to data/candidates/candidates.jsonl. No verification
happens here — that is bench_core.verify's job.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from bench_core.domain import Candidate
from bench_core.storage import CANDIDATES, read_jsonl, write_jsonl
from bench_core.goldgen.base import (
    DEFAULT_FORM,
    QUERY_FORMS,
    RegistryAdapter,
    build_variants,
    since_default,
    strip_token,
    utcnow,
)
from bench_core.goldgen.fed_register import FederalRegisterAdapter
from bench_core.goldgen.github_releases import GithubReleasesAdapter
from bench_core.goldgen.nvd_cve import NvdAdapter
from bench_core.goldgen.sec_edgar import SecEdgarAdapter

ADAPTERS = {
    "fed_register": FederalRegisterAdapter,
    "nvd_cve": NvdAdapter,
    "github_releases": GithubReleasesAdapter,
    "sec_edgar": SecEdgarAdapter,
}

CANDIDATES_FILE = CANDIDATES / "candidates.jsonl"

__all__ = [
    "ADAPTERS",
    "FederalRegisterAdapter",
    "GithubReleasesAdapter",
    "NvdAdapter",
    "SecEdgarAdapter",
    "RegistryAdapter",
    "build_variants",
    "strip_token",
    "since_default",
    "utcnow",
    "QUERY_FORMS",
    "DEFAULT_FORM",
    "run",
    "run_sync",
]


def _existing_ids() -> set[str]:
    return {r["row_id"] for r in read_jsonl(CANDIDATES_FILE)}


async def run(
    source: str = "all", since: datetime | None = None, limit: int = 100, days: int = 2
) -> int:
    since = since or since_default(days)
    names = list(ADAPTERS) if source == "all" else [source]
    seen = _existing_ids()
    fresh: list[Candidate] = []
    for name in names:
        adapter = ADAPTERS[name]()
        try:
            cands = await adapter.fetch(since, limit)
        except Exception as exc:  # one source failing must not sink the batch
            print(f"[pump] {name} failed: {exc!r}")
            continue
        for c in cands:
            if c.row_id not in seen:
                seen.add(c.row_id)
                fresh.append(c)
        print(f"[pump] {name}: {len(cands)} fetched")
    if fresh:
        write_jsonl(CANDIDATES_FILE, (c.to_dict() for c in fresh), append=True)
    print(f"[pump] {len(fresh)} new candidates appended")
    return len(fresh)


def run_sync(source: str = "all", limit: int = 100, days: int = 2) -> int:
    return asyncio.run(run(source=source, limit=limit, days=days))
