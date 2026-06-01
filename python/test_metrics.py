"""Unit tests for sim_utils metric helpers.

Tests cover: nrmse_distribution, importance_weighted_nrmse, mean_calibration.
"""

import pytest

from sim_utils import importance_weighted_nrmse, mean_calibration, nrmse_distribution


# ---------------------------------------------------------------------------
# Fixtures: hand-constructed eval_json dicts
# ---------------------------------------------------------------------------

NORMAL_EVAL = {
    "mean_nrmse": 0.1,
    "mean_coverage": 0.9,
    "mean_calibration": 0.85,
    "events_with_zero_coverage": 0,
    "per_event": [
        {"event": "cycles", "nrmse": 0.05, "coverage": 0.95, "gt_cv": 0.2, "calibration": 0.9},
        {"event": "instructions", "nrmse": 0.10, "coverage": 0.90, "gt_cv": 0.5, "calibration": 0.8},
        {"event": "cache-misses", "nrmse": 0.40, "coverage": 0.80, "gt_cv": 1.2, "calibration": 0.85},
    ],
}

ALL_NULL_NRMSE = {
    "mean_nrmse": None,
    "mean_coverage": None,
    "mean_calibration": None,
    "events_with_zero_coverage": 3,
    "per_event": [
        {"event": "cycles", "nrmse": None, "coverage": 0.0, "gt_cv": 0.2, "calibration": None},
        {"event": "instructions", "nrmse": None, "coverage": 0.0, "gt_cv": 0.5, "calibration": None},
    ],
}

NULL_GT_CV = {
    "mean_nrmse": 0.2,
    "mean_coverage": 0.85,
    "mean_calibration": 0.7,
    "events_with_zero_coverage": 0,
    "per_event": [
        {"event": "cycles", "nrmse": 0.1, "coverage": 0.9, "gt_cv": None, "calibration": 0.7},
        {"event": "instructions", "nrmse": 0.3, "coverage": 0.8, "gt_cv": 0.0, "calibration": 0.7},
    ],
}

ZERO_GT_CV_MIXED = {
    "mean_nrmse": 0.15,
    "mean_coverage": 0.88,
    "mean_calibration": 0.75,
    "events_with_zero_coverage": 0,
    "per_event": [
        {"event": "cycles", "nrmse": 0.1, "coverage": 0.9, "gt_cv": 0.0, "calibration": 0.8},
        {"event": "cache-misses", "nrmse": 0.2, "coverage": 0.85, "gt_cv": 0.8, "calibration": 0.7},
    ],
}

EMPTY_PER_EVENT = {
    "mean_nrmse": None,
    "mean_coverage": None,
    "mean_calibration": None,
    "events_with_zero_coverage": 0,
    "per_event": [],
}

NO_PER_EVENT_KEY = {
    "mean_nrmse": None,
    "mean_coverage": None,
    "mean_calibration": 0.5,
}


# ---------------------------------------------------------------------------
# Tests: nrmse_distribution
# ---------------------------------------------------------------------------


class TestNrmseDistribution:
    def test_normal_case_keys(self):
        result = nrmse_distribution(NORMAL_EVAL)
        assert set(result.keys()) == {"mean", "p50", "p90", "max"}

    def test_normal_case_values_are_floats(self):
        result = nrmse_distribution(NORMAL_EVAL)
        for key in ("mean", "p50", "p90", "max"):
            assert isinstance(result[key], float), f"{key} should be float"

    def test_normal_case_ordering(self):
        # mean, p50, p90, max should be non-decreasing for a reasonable distribution
        result = nrmse_distribution(NORMAL_EVAL)
        assert result["p50"] <= result["p90"] <= result["max"]

    def test_normal_case_max_is_largest(self):
        result = nrmse_distribution(NORMAL_EVAL)
        assert result["max"] == pytest.approx(0.40)

    def test_normal_case_p90_between_p50_and_max(self):
        result = nrmse_distribution(NORMAL_EVAL)
        assert result["p50"] <= result["p90"] <= result["max"]

    def test_all_null_nrmse_returns_none_values(self):
        result = nrmse_distribution(ALL_NULL_NRMSE)
        assert result == {"mean": None, "p50": None, "p90": None, "max": None}

    def test_empty_per_event_returns_none_values(self):
        result = nrmse_distribution(EMPTY_PER_EVENT)
        assert result == {"mean": None, "p50": None, "p90": None, "max": None}

    def test_missing_per_event_key_returns_none_values(self):
        result = nrmse_distribution(NO_PER_EVENT_KEY)
        assert result == {"mean": None, "p50": None, "p90": None, "max": None}

    def test_single_event(self):
        single = {"per_event": [{"nrmse": 0.3, "gt_cv": 0.5}]}
        result = nrmse_distribution(single)
        assert result["mean"] == pytest.approx(0.3)
        assert result["p50"] == pytest.approx(0.3)
        assert result["p90"] == pytest.approx(0.3)
        assert result["max"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Tests: importance_weighted_nrmse
# ---------------------------------------------------------------------------


class TestImportanceWeightedNrmse:
    def test_normal_case_returns_float(self):
        result = importance_weighted_nrmse(NORMAL_EVAL)
        assert isinstance(result, float)

    def test_normal_case_weighted_toward_high_cv(self):
        # cache-misses has nrmse=0.40 and gt_cv=1.2 (highest weight)
        # weighted result should be pulled toward 0.40 vs unweighted mean 0.183
        result = importance_weighted_nrmse(NORMAL_EVAL)
        unweighted_mean = (0.05 + 0.10 + 0.40) / 3
        assert result > unweighted_mean, "weighted nRMSE should exceed simple mean"

    def test_normal_case_manual_calculation(self):
        # cycles: nrmse=0.05, gt_cv=0.2  -> contribution: 0.2*0.05=0.010
        # instructions: nrmse=0.10, gt_cv=0.5 -> contribution: 0.5*0.10=0.050
        # cache-misses: nrmse=0.40, gt_cv=1.2 -> contribution: 1.2*0.40=0.480
        # total weight = 0.2+0.5+1.2 = 1.9
        # weighted mean = 0.540 / 1.9
        expected = (0.2 * 0.05 + 0.5 * 0.10 + 1.2 * 0.40) / (0.2 + 0.5 + 1.2)
        result = importance_weighted_nrmse(NORMAL_EVAL)
        assert result == pytest.approx(expected)

    def test_all_null_nrmse_returns_none(self):
        assert importance_weighted_nrmse(ALL_NULL_NRMSE) is None

    def test_null_gt_cv_skipped(self):
        # Both events in NULL_GT_CV have null or zero gt_cv -> no valid events
        assert importance_weighted_nrmse(NULL_GT_CV) is None

    def test_zero_gt_cv_skipped_nonzero_used(self):
        # ZERO_GT_CV_MIXED: cycles has gt_cv=0.0 (skipped), cache-misses has gt_cv=0.8
        result = importance_weighted_nrmse(ZERO_GT_CV_MIXED)
        assert result is not None
        assert result == pytest.approx(0.2)  # only cache-misses contributes: 0.8*0.2/0.8

    def test_empty_per_event_returns_none(self):
        assert importance_weighted_nrmse(EMPTY_PER_EVENT) is None

    def test_missing_per_event_key_returns_none(self):
        assert importance_weighted_nrmse(NO_PER_EVENT_KEY) is None


# ---------------------------------------------------------------------------
# Tests: mean_calibration
# ---------------------------------------------------------------------------


class TestMeanCalibration:
    def test_normal_case_returns_headline_value(self):
        result = mean_calibration(NORMAL_EVAL)
        assert result == pytest.approx(0.85)

    def test_null_calibration_returns_none(self):
        assert mean_calibration(ALL_NULL_NRMSE) is None

    def test_missing_key_returns_none(self):
        assert mean_calibration({}) is None

    def test_no_per_event_key_returns_headline(self):
        # mean_calibration only reads the headline, not per_event
        result = mean_calibration(NO_PER_EVENT_KEY)
        assert result == pytest.approx(0.5)

    def test_zero_calibration_returns_zero(self):
        result = mean_calibration({"mean_calibration": 0.0})
        assert result == pytest.approx(0.0)
