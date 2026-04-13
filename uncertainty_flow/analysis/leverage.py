"""Feature leverage analysis for uncertainty attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel


def _interval_indices(prediction, confidence: float) -> tuple[int, int]:
    """Return lower/upper quantile indices for a confidence level."""
    alpha = (1 - confidence) / 2
    lower_idx = prediction._find_nearest_quantile_index(alpha)
    upper_idx = prediction._find_nearest_quantile_index(1 - alpha)
    return lower_idx, upper_idx


def _point_matrix(prediction) -> np.ndarray:
    """Return point predictions as a 2D array with one column per target."""
    median_idx = prediction._find_nearest_quantile_index(0.5)

    if len(prediction._targets) == 1:
        return prediction._quantiles[:, median_idx].reshape(-1, 1)

    columns = [
        prediction._quantiles[:, target_idx * prediction._n_quantiles + median_idx]
        for target_idx in range(len(prediction._targets))
    ]
    return np.column_stack(columns)


def _interval_width_matrix(prediction, confidence: float) -> np.ndarray:
    """Return interval widths as a 2D array with one column per target."""
    lower_idx, upper_idx = _interval_indices(prediction, confidence)

    if len(prediction._targets) == 1:
        widths = prediction._quantiles[:, upper_idx] - prediction._quantiles[:, lower_idx]
        return widths.reshape(-1, 1)

    width_columns = []
    for target_idx in range(len(prediction._targets)):
        base_idx = target_idx * prediction._n_quantiles
        lower = prediction._quantiles[:, base_idx + lower_idx]
        upper = prediction._quantiles[:, base_idx + upper_idx]
        width_columns.append(upper - lower)

    return np.column_stack(width_columns)


def _interval_widths(prediction, confidence: float, target_name: str | None = None) -> np.ndarray:
    """Return interval widths for a target."""
    widths = _interval_width_matrix(prediction, confidence)
    if widths.shape[1] == 1:
        return widths[:, 0]

    selected_target = target_name or prediction._targets[0]
    target_idx = prediction._targets.index(selected_target)
    return widths[:, target_idx]


def _joint_interval_scale(width_matrix: np.ndarray) -> np.ndarray:
    """Return a normalized joint interval scale across targets."""
    stabilized = np.clip(width_matrix, 1e-12, None)
    return np.exp(np.mean(np.log(stabilized), axis=1))


def _rank_correlation_matrix(point_matrix: np.ndarray) -> np.ndarray:
    """Return a stable rank-correlation matrix across targets."""
    if point_matrix.shape[1] <= 1:
        return np.eye(point_matrix.shape[1], dtype=float)

    ranks = np.column_stack(
        [
            pl.Series(point_matrix[:, idx]).rank(method="average").to_numpy()
            for idx in range(point_matrix.shape[1])
        ]
    )
    corr = np.corrcoef(ranks.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _mean_upper_triangle_abs(matrix: np.ndarray) -> float:
    """Return the mean absolute value over the upper triangle, excluding the diagonal."""
    if matrix.shape[0] <= 1:
        return 0.0

    tri_rows, tri_cols = np.triu_indices(matrix.shape[0], k=1)
    values = np.abs(matrix[tri_rows, tri_cols])
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _generate_recommendation(
    aleatoric_score: float,
    epistemic_score: float,
    leverage_score: float,
    leverage_threshold: float = 0.5,
    aleatoric_dominance_ratio: float = 2.0,
) -> str:
    """
    Generate actionable recommendation based on uncertainty decomposition.

    Args:
        aleatoric_score: Irreducible uncertainty contribution
        epistemic_score: Reducible uncertainty contribution
        leverage_score: Total impact on prediction intervals
        leverage_threshold: Threshold for "high leverage" classification
        aleatoric_dominance_ratio: Ratio for aleatoric-dominant classification

    Returns:
        Recommendation string
    """
    # High aleatoric, low epistemic → Accept uncertainty (inherently noisy)
    if aleatoric_score > epistemic_score * aleatoric_dominance_ratio:
        return "accept_uncertainty"

    # High epistemic, low aleatoric → Collect more training data
    if epistemic_score > aleatoric_score:
        return "collect_more_data"

    # High leverage overall → Prioritize accurate measurement
    if leverage_score > leverage_threshold:
        return "high_leverage"

    # Low leverage overall → Can approximate for efficiency
    return "low_leverage"


def _format_recommendation(rec_type: str) -> str:
    """Format recommendation type into human-readable string."""
    recommendations = {
        "accept_uncertainty": "Accept uncertainty (inherently noisy)",
        "collect_more_data": "Collect more training data",
        "high_leverage": "High leverage - prioritize accurate measurement",
        "low_leverage": "Low leverage - can approximate for efficiency",
    }
    return recommendations.get(rec_type, "Unknown")


class FeatureLeverageAnalyzer:
    """
    Analyze which features most influence prediction uncertainty.

    Identifies leverage features by decomposing uncertainty into aleatoric
    (irreducible) and epistemic (reducible) components and computing how
    much each feature contributes to prediction interval width.

    Parameters
    ----------
    model : BaseUncertaintyModel
        Fitted uncertainty model with predict() method
    confidence : float, default=0.9
        Confidence level for prediction intervals
    n_perturbations : int, default=100
        Number of perturbation samples for leverage estimation
    n_bins : int, default=10
        Number of quantile bins for conditional decomposition
    leverage_threshold : float, default=0.5
        Threshold for "high leverage" classification
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> import polars as pl
    >>> from uncertainty_flow.models import QuantileForestForecaster
    >>> from uncertainty_flow.analysis import FeatureLeverageAnalyzer
    >>>
    >>> # Train forecaster
    >>> model = QuantileForestForecaster(targets="demand", horizon=7)
    >>> model.fit(train_data)
    >>>
    >>> # Identify leverage features
    >>> analyzer = FeatureLeverageAnalyzer(model)
    >>> report = analyzer.analyze(test_data)
    >>>
    >>> # Filter to high-leverage features
    >>> high_leverage = report.filter(pl.col("leverage_score") > 0.5)
    >>> print(high_leverage["feature"].to_list())
    """

    def __init__(
        self,
        model: "BaseUncertaintyModel",
        confidence: float = 0.9,
        n_perturbations: int = 100,
        n_bins: int = 10,
        leverage_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        self.model = model
        self.confidence = confidence
        self.n_perturbations = n_perturbations
        self.n_bins = n_bins
        self.leverage_threshold = leverage_threshold
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._prediction_row_budget = 800

    def analyze(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Analyze feature leverage on prediction uncertainty.

        Computes leverage scores, aleatoric/epistemic decomposition,
        and actionable recommendations for each feature.

        Args:
            data: Feature DataFrame for leverage analysis

        Returns:
            Polars DataFrame with columns:
                - feature: Feature name
                - aleatoric_score: Irreducible uncertainty contribution
                - epistemic_score: Reducible uncertainty contribution
                - leverage_score: Total impact on prediction intervals
                - recommendation: Actionable insight

        Raises:
            InvalidDataError: If data is empty or contains invalid data
        """
        from ..utils.exceptions import error_invalid_data

        if data.height == 0:
            error_invalid_data("Cannot analyze leverage on empty DataFrame")

        baseline_pred = self.model.predict(data)
        baseline_width_matrix = _interval_width_matrix(baseline_pred, self.confidence)
        baseline_width = baseline_width_matrix[:, 0]
        n_repeats = self._effective_perturbation_count(data.height)

        results = []

        for feature_name in data.columns:
            feature_vals = data[feature_name].to_numpy()

            # Skip constant features
            if np.std(feature_vals) == 0:
                continue

            # Compute leverage score via permutation
            perturbed_width_stack, _ = self._predict_perturbation_effects(
                data, feature_name, n_repeats
            )
            leverage_score = self._compute_permutation_leverage(
                baseline_width,
                perturbed_width_stack[:, :, 0],
            )

            # Compute aleatoric/epistemic decomposition
            aleatoric_score, epistemic_score = self._compute_decomposition(
                feature_vals, baseline_width
            )

            # Generate recommendation
            rec_type = _generate_recommendation(
                aleatoric_score, epistemic_score, leverage_score, self.leverage_threshold
            )
            recommendation = _format_recommendation(rec_type)

            results.append(
                {
                    "feature": feature_name,
                    "aleatoric_score": float(aleatoric_score),
                    "epistemic_score": float(epistemic_score),
                    "leverage_score": float(leverage_score),
                    "recommendation": recommendation,
                }
            )

        if not results:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "aleatoric_score": pl.Float64,
                    "epistemic_score": pl.Float64,
                    "leverage_score": pl.Float64,
                    "recommendation": pl.String,
                }
            )

        df = pl.DataFrame(results)
        return df.sort("leverage_score", descending=True)

    def _compute_permutation_leverage(
        self,
        baseline_width: np.ndarray,
        perturbed_widths: np.ndarray,
    ) -> float:
        """
        Compute leverage score via feature permutation.

        Measures how much prediction interval width changes when
        the feature is randomly permuted (breaking its relationship
        with the target).

        Args:
            baseline_width: Baseline prediction interval widths
            perturbed_widths: Interval widths with shape (n_perturbations, n_rows)

        Returns:
            Leverage score (absolute change in interval width)
        """
        width_deltas = np.abs(perturbed_widths - baseline_width.reshape(1, -1))
        return float(width_deltas.mean())

    def _effective_perturbation_count(self, n_rows: int) -> int:
        """
        Return a bounded perturbation count based on requested repeats and frame size.

        The analyzer keeps total perturbed prediction rows within a lightweight budget
        so leverage analysis stays practical on larger evaluation frames.
        """
        if n_rows <= 0:
            return 1

        max_repeats_by_rows = max(1, self._prediction_row_budget // n_rows)
        return max(1, min(self.n_perturbations, max_repeats_by_rows))

    def _predict_perturbation_effects(
        self,
        data: pl.DataFrame,
        feature_name: str,
        n_repeats: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict all perturbation repeats for one feature in a single batched call."""
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive")

        frames = []
        feature_vals = data[feature_name].to_numpy()

        for _ in range(n_repeats):
            permuted_vals = self._rng.permutation(feature_vals)
            frames.append(data.with_columns(pl.Series(feature_name, permuted_vals)))

        batched = pl.concat(frames, rechunk=False)
        perturbed_pred = self.model.predict(batched)
        width_matrix = _interval_width_matrix(perturbed_pred, self.confidence)
        point_matrix = _point_matrix(perturbed_pred)

        n_rows = data.height
        n_targets = width_matrix.shape[1]
        reshaped_widths = width_matrix.reshape(n_repeats, n_rows, n_targets)
        reshaped_points = point_matrix.reshape(n_repeats, n_rows, point_matrix.shape[1])
        return reshaped_widths, reshaped_points

    def _compute_joint_leverage(
        self,
        baseline_width_matrix: np.ndarray,
        baseline_point_matrix: np.ndarray,
        perturbed_width_stack: np.ndarray,
        perturbed_point_stack: np.ndarray,
    ) -> tuple[float, float]:
        """Compute joint leverage and dependence shift across targets."""
        baseline_joint_scale = _joint_interval_scale(baseline_width_matrix)
        perturbed_joint_scales = np.stack(
            [_joint_interval_scale(width_matrix) for width_matrix in perturbed_width_stack],
            axis=0,
        )
        volume_shift = np.abs(perturbed_joint_scales - baseline_joint_scale.reshape(1, -1)).mean()

        baseline_corr = _rank_correlation_matrix(baseline_point_matrix)
        dependence_shifts = []
        for point_matrix in perturbed_point_stack:
            corr_delta = _rank_correlation_matrix(point_matrix) - baseline_corr
            dependence_shifts.append(_mean_upper_triangle_abs(corr_delta))

        dependence_shift = float(np.mean(dependence_shifts)) if dependence_shifts else 0.0
        return float(volume_shift + dependence_shift), dependence_shift

    def _compute_decomposition(
        self,
        feature_vals: np.ndarray,
        baseline_width: np.ndarray,
    ) -> tuple[float, float]:
        """
        Decompose uncertainty into aleatoric and epistemic components.

        Uses conditional variance decomposition:
        - Aleatoric: Mean of within-group variance (noise within bins)
        - Epistemic: Variance of between-group means (model uncertainty)

        Args:
            feature_vals: Feature values
            baseline_width: Prediction interval widths

        Returns:
            Tuple of (aleatoric_score, epistemic_score)
        """
        # Bin feature values into quantiles
        try:
            # Create Polars Series for qcut
            feature_series = pl.Series(feature_vals)
            binned = feature_series.qcut(self.n_bins, allow_duplicates=True)
            bin_labels = binned.to_numpy()
        except (ValueError, pl.ColumnNotFoundError, Exception):
            # Fallback: use equal-width bins if qcut fails
            bin_edges = np.linspace(np.min(feature_vals), np.max(feature_vals), self.n_bins + 1)
            bin_labels = np.digitize(feature_vals, bin_edges[:-1])

        # Compute within-group and between-group variance
        unique_bins = np.unique(bin_labels)

        if len(unique_bins) <= 1:
            # Only one bin, cannot decompose
            return 0.0, 0.0

        within_group_vars = []
        between_group_means = []

        for bin_label in unique_bins:
            bin_mask = bin_labels == bin_label
            bin_widths = baseline_width[bin_mask]

            if len(bin_widths) > 1:
                within_group_vars.append(np.var(bin_widths))
                between_group_means.append(np.mean(bin_widths))

        if not within_group_vars:
            return 0.0, 0.0

        # Aleatoric: mean of within-group variances
        aleatoric_score = np.mean(within_group_vars)

        # Epistemic: variance of between-group means
        epistemic_score = np.var(between_group_means) if len(between_group_means) > 1 else 0.0

        return aleatoric_score, epistemic_score

    def analyze_multivariate(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Analyze feature leverage for multivariate forecasting models.

        Extends leverage analysis to multiple targets, computing
        per-target and joint leverage scores.

        Args:
            data: Feature DataFrame for leverage analysis

        Returns:
            Polars DataFrame with columns:
                - feature: Feature name
                - target: Target name (or "joint" for multivariate impact)
                - aleatoric_score: Irreducible uncertainty contribution
                - epistemic_score: Reducible uncertainty contribution
                - leverage_score: Total impact on prediction intervals
                - dependence_shift: Rank-dependence change across targets
                - recommendation: Actionable insight
        """
        pred = self.model.predict(data)

        if len(pred._targets) <= 1:
            # Not multivariate, fall back to standard analysis
            return self.analyze(data)

        baseline_width_matrix = _interval_width_matrix(pred, self.confidence)
        baseline_point_matrix = _point_matrix(pred)
        target_names = list(pred._targets)
        n_repeats = self._effective_perturbation_count(data.height)
        results = []

        for feature_name in data.columns:
            feature_vals = data[feature_name].to_numpy()
            if np.std(feature_vals) == 0:
                continue

            perturbed_width_stack, perturbed_point_stack = self._predict_perturbation_effects(
                data,
                feature_name,
                n_repeats,
            )

            for target_idx, target_name in enumerate(target_names):
                baseline_width = baseline_width_matrix[:, target_idx]
                leverage_score = self._compute_permutation_leverage(
                    baseline_width,
                    perturbed_width_stack[:, :, target_idx],
                )
                aleatoric_score, epistemic_score = self._compute_decomposition(
                    feature_vals,
                    baseline_width,
                )
                recommendation = _format_recommendation(
                    _generate_recommendation(
                        aleatoric_score,
                        epistemic_score,
                        leverage_score,
                        self.leverage_threshold,
                    )
                )
                results.append(
                    {
                        "feature": feature_name,
                        "target": target_name,
                        "aleatoric_score": float(aleatoric_score),
                        "epistemic_score": float(epistemic_score),
                        "leverage_score": float(leverage_score),
                        "dependence_shift": 0.0,
                        "recommendation": recommendation,
                    }
                )

            joint_width = _joint_interval_scale(baseline_width_matrix)
            joint_leverage, dependence_shift = self._compute_joint_leverage(
                baseline_width_matrix,
                baseline_point_matrix,
                perturbed_width_stack,
                perturbed_point_stack,
            )
            joint_aleatoric, joint_epistemic = self._compute_decomposition(
                feature_vals,
                joint_width,
            )
            joint_recommendation = _format_recommendation(
                _generate_recommendation(
                    joint_aleatoric,
                    joint_epistemic,
                    joint_leverage,
                    self.leverage_threshold,
                )
            )
            results.append(
                {
                    "feature": feature_name,
                    "target": "joint",
                    "aleatoric_score": float(joint_aleatoric),
                    "epistemic_score": float(joint_epistemic),
                    "leverage_score": float(joint_leverage),
                    "dependence_shift": float(dependence_shift),
                    "recommendation": joint_recommendation,
                }
            )

        if results:
            return pl.DataFrame(results).sort(
                ["target", "leverage_score"],
                descending=[False, True],
            )

        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "target": pl.String,
                "aleatoric_score": pl.Float64,
                "epistemic_score": pl.Float64,
                "leverage_score": pl.Float64,
                "dependence_shift": pl.Float64,
                "recommendation": pl.String,
            }
        )

    def summary(self) -> dict[str, Any]:
        """
        Return summary of the analyzer configuration.

        Returns:
            Dictionary with analyzer configuration
        """
        return {
            "confidence": self.confidence,
            "n_perturbations": self.n_perturbations,
            "n_bins": self.n_bins,
            "leverage_threshold": self.leverage_threshold,
            "random_state": self.random_state,
            "effective_prediction_row_budget": self._prediction_row_budget,
        }
