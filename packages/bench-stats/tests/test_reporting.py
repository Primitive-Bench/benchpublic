"""Tests for the richer reporting stats (McNemar pair, power, sizing, CMH, CUSUM,
tied-rank band). Adapted from arlenk2021/GoldenEvalsWebSearch tests/test_stats.py.

These depend on scipy/statsmodels; they are skipped if those are not installed.
"""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from bench_stats import (  # noqa: E402
    cmh_global,
    cusum,
    mcnemar_pair,
    mcnemar_power,
    required_n,
    tied_rank_band,
)
from bench_stats.reporting import RankBand, discordance_rate  # noqa: E402


def test_mcnemar_symmetric_is_nonsignificant():
    _, p = mcnemar_pair(10, 10)
    assert p > 0.5


def test_mcnemar_lopsided_is_significant():
    _, p = mcnemar_pair(20, 2)
    assert p < 0.05


def test_mcnemar_exact_below_cutoff():
    # b + c = 22 < 25 -> exact binomial path, still returns a valid p in [0,1]
    _, p = mcnemar_pair(2, 20)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_required_n_grows_with_discordance():
    # For a FIXED marginal gap, McNemar n scales as discordance/gap^2:
    # higher discordance (vendors disagree more) => larger n. n=500 powers a
    # ~3.6pp gap only when discordance is LOW (<~0.08).
    gap = 0.036
    low_disc = required_n((0.08 + gap) / 2, (0.08 - gap) / 2)  # discordance 0.08
    high_disc = required_n((0.40 + gap) / 2, (0.40 - gap) / 2)  # discordance 0.40
    assert low_disc < high_disc
    assert low_disc <= 500 < high_disc
    assert mcnemar_power(low_disc, (0.08 + gap) / 2, (0.08 - gap) / 2) >= 0.79


def test_required_n_infinite_when_no_gap():
    assert required_n(0.1, 0.1) == float("inf")


def test_mcnemar_power_zero_discordance():
    assert mcnemar_power(100, 0.0, 0.0) == 0.0


def test_mcnemar_power_increases_with_n():
    lo = mcnemar_power(100, 0.06, 0.02)
    hi = mcnemar_power(400, 0.06, 0.02)
    assert hi > lo


def test_discordance_rate():
    assert discordance_rate(3, 5, 100) == pytest.approx(0.08)
    assert discordance_rate(0, 0, 0) == 0.0


def test_cmh_global_runs_across_strata():
    # Two strata, both leaning the same direction -> small p-value.
    tables = [
        [[30, 10], [10, 30]],
        [[25, 8], [9, 28]],
    ]
    stat, p = cmh_global(tables)
    assert stat >= 0
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_cusum_accumulates_positive_drift():
    out = cusum([1.0, 1.0, 1.0, 1.0], target=0.0, k=0.0)
    assert out == [1.0, 2.0, 3.0, 4.0]


def test_cusum_resets_on_negative():
    out = cusum([-5.0, -5.0, 2.0], target=0.0, k=0.0)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 2.0


def test_tied_rank_band_clear_winner():
    # Leader's lower bound clears the runner-up's upper bound -> single winner.
    rates = [
        ("A", 0.90, 0.85, 0.95),
        ("B", 0.50, 0.40, 0.60),
    ]
    rb = tied_rank_band(rates)
    assert isinstance(rb, RankBand)
    assert rb.winner == "A"
    assert rb.band == ["A"]


def test_tied_rank_band_overlap_is_tie():
    rates = [
        ("A", 0.62, 0.50, 0.74),
        ("B", 0.58, 0.46, 0.70),
        ("C", 0.20, 0.10, 0.30),
    ]
    rb = tied_rank_band(rates)
    assert rb.winner is None
    assert set(rb.band) == {"A", "B"}


def test_tied_rank_band_unsorted_input():
    rates = [
        ("B", 0.50, 0.40, 0.60),
        ("A", 0.90, 0.85, 0.95),
    ]
    rb = tied_rank_band(rates)
    assert rb.band[0] == "A"
    assert rb.winner == "A"


def test_tied_rank_band_empty():
    rb = tied_rank_band([])
    assert rb.band == []
    assert rb.winner is None
