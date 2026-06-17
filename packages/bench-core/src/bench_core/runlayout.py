"""Per-run result directory layout (ann-benchmarks pattern).

    runs/<run_id>/
        manifest.json     # RunManifest
        items.jsonl       # one ItemResult per line (streamed during the run)
        slices.jsonl      # SliceResult records (derived by bench-stats, re-emitted)

This directory is the unit the ingest pipeline (platform/ingest) pulls into DuckDB.
Compressible, enumerable, re-runnable. Held-out runs write the same layout but
their items.jsonl never leaves the private eval server.
"""

from __future__ import annotations

from pathlib import Path

from bench_schemas import ItemResult, RunManifest, SliceResult


class RunDir:
    def __init__(self, root: str | Path, run_id: str):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: RunManifest) -> None:
        (self.path / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    def append_item(self, item: ItemResult) -> None:
        with (self.path / "items.jsonl").open("a") as f:
            f.write(item.model_dump_json() + "\n")

    def write_slices(self, slices: list[SliceResult]) -> None:
        with (self.path / "slices.jsonl").open("w") as f:
            for s in slices:
                f.write(s.model_dump_json() + "\n")
