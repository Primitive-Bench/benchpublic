"""Contract smoke tests. These pin the v0.1.0 surface so accidental breaks fail CI."""

from datetime import datetime

from bench_schemas import (
    AdapterSpec,
    ItemResult,
    RunManifest,
    ScorerOutput,
    SliceResult,
    SCHEMA_VERSION,
)
from bench_schemas.models import Primitive


def test_schema_version_pinned():
    assert SCHEMA_VERSION == "0.1.0"


def test_round_trip_item_result():
    item = ItemResult(
        run_id="run_1",
        adapter="claude-sonnet-ocr",
        item_id="doc_42",
        primitive=Primitive.OCR,
        slices=["doc_type:invoice", "lang:en"],
        output=ScorerOutput(correct=True, metrics={"cer": 0.01}),
    )
    assert ItemResult.model_validate_json(item.model_dump_json()) == item


def test_manifest_requires_seed_and_versions():
    m = RunManifest(
        run_id="run_1",
        primitive=Primitive.OCR,
        created_at=datetime(2026, 6, 16),
        seed=7,
        adapters=[
            AdapterSpec(
                name="tesseract",
                primitive=Primitive.OCR,
                vendor="oss",
                version="5.3",
                is_sentinel=True,
            )
        ],
        task_version="ocr@1",
        dataset_version="ocr-2026.06",
    )
    assert m.split == "public_dev"
    assert m.schema_version == SCHEMA_VERSION


def test_slice_result_separability_default_none():
    s = SliceResult(
        run_id="run_1",
        primitive=Primitive.OCR,
        slice="doc_type:invoice",
        adapter="tesseract",
        n=120,
        point_estimate=0.91,
    )
    assert s.separable is None  # must be explicitly set by bench-stats
