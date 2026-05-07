"""Tests for log-score metric."""

import numpy as np
import pytest
from scipy import stats

from uncertainty_flow.metrics.log_score import log_score, log_score_kde, log_score_pooled


class TestLogScore:
    def test_normal_perfect_score_high(self):
        levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        y = np.array([0.0])
        q_matrix = np.tile(qv, (1, 1))
        score = log_score(y, q_matrix, levels, family="normal")
        assert np.isfinite(score)
        assert score < 0

    def test_worse_prediction_lower_score(self):
        levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        good_qv = stats.norm.ppf(levels, loc=0, scale=1)
        bad_qv = stats.norm.ppf(levels, loc=10, scale=1)
        y = np.array([0.0])
        good_score = log_score(y, np.tile(good_qv, (1, 1)), levels, family="normal")
        bad_score = log_score(y, np.tile(bad_qv, (1, 1)), levels, family="normal")
        assert good_score > bad_score

    def test_auto_family(self):
        levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        qv = stats.norm.ppf(levels, loc=5, scale=2)
        y = np.array([5.0, 4.5, 5.5])
        q_matrix = np.tile(qv, (3, 1))
        score = log_score(y, q_matrix, levels, family="auto")
        assert np.isfinite(score)

    def test_batch_mean(self):
        levels = np.array([0.1, 0.5, 0.9])
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        y = np.array([0.0, 1.0, -1.0])
        q_matrix = np.tile(qv, (3, 1))
        score = log_score(y, q_matrix, levels)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_rowwise_beats_pooled_on_heteroscedastic_data(self):
        levels = np.array([0.1, 0.5, 0.9])
        y = np.array([0.0, 5.0])
        row1 = stats.norm.ppf(levels, loc=0.0, scale=0.5)
        row2 = stats.norm.ppf(levels, loc=5.0, scale=2.0)
        q_matrix = np.vstack([row1, row2])

        rowwise = log_score(y, q_matrix, levels, family="normal")
        pooled = log_score_pooled(y, q_matrix, levels, family="normal")
        assert rowwise > pooled


class TestLogScoreKDE:
    def test_basic_kde(self):
        levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        y = np.array([0.0])
        q_matrix = np.tile(qv, (1, 1))
        score = log_score_kde(y, q_matrix, levels, n_draw=200, random_state=42)
        assert np.isfinite(score)
        assert score < 0

    def test_kde_reproducible(self):
        levels = np.array([0.1, 0.5, 0.9])
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        y = np.array([0.0])
        q_matrix = np.tile(qv, (1, 1))
        s1 = log_score_kde(y, q_matrix, levels, n_draw=100, random_state=42)
        s2 = log_score_kde(y, q_matrix, levels, n_draw=100, random_state=42)
        assert s1 == pytest.approx(s2)


class TestLogScoreConvenience:
    def test_distribution_prediction_log_score(self):
        from uncertainty_flow.core.distribution import DistributionPrediction

        levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        qv = stats.norm.ppf(levels, loc=0, scale=1)
        q_matrix = np.tile(qv, (10, 1))
        pred = DistributionPrediction(q_matrix, levels, target_names=["y"])
        y = np.zeros(10)
        score = pred.log_score(y, family="normal")
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_multivariate_log_score(self):
        from uncertainty_flow.core.distribution import DistributionPrediction

        levels = [0.1, 0.5, 0.9]
        qv1 = stats.norm.ppf(levels, loc=0, scale=1)
        qv2 = stats.norm.ppf(levels, loc=5, scale=2)
        q_matrix = np.column_stack([np.tile(qv1, (5, 1)), np.tile(qv2, (5, 1))])
        pred = DistributionPrediction(q_matrix, levels, target_names=["a", "b"])
        y = np.column_stack([np.zeros(5), np.full(5, 5.0)])
        result = pred.log_score(y, family="normal")
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result
