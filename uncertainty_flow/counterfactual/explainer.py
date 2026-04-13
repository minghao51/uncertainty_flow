"""Counterfactual explanations for uncertainty reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from .search import EvolutionarySearch, GradientSearch, SearchResult

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel


class UncertaintyExplainer:
    """
    Explain uncertainty by finding minimal feature changes to reduce intervals.

    Answers "what would need to change about this input for us to be more confident?"
    by searching for counterfactual examples that achieve target reduction in
    prediction interval width with minimal feature perturbations.

    Parameters
    ----------
    model : BaseUncertaintyModel
        Fitted uncertainty model with predict() method
    confidence : float, default=0.9
        Confidence level for prediction intervals
    method : {"auto", "evolutionary", "gradient"}, default="auto"
        Search strategy:
        - "auto": Automatically choose based on model type
        - "evolutionary": Genetic algorithm (tree-based models)
        - "gradient": Gradient-based (differentiable models)
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> import polars as pl
    >>> from uncertainty_flow.models import QuantileForestForecaster
    >>> from uncertainty_flow.counterfactual import UncertaintyExplainer
    >>>
    >>> # Train model
    >>> model = QuantileForestForecaster(targets="demand", horizon=7)
    >>> model.fit(train_data)
    >>>
    >>> # Explain uncertainty for a prediction
    >>> explainer = UncertaintyExplainer(model, random_state=42)
    >>> result = explainer.explain_uncertainty(
    ...     X_test.head(1),
    ...     target_reduction=0.5,
    ...     feature_bounds={"temperature": (0, 40), "humidity": (0, 100)}
    ... )
    >>>
    >>> # View counterfactual explanation
    >>> print(result.to_polars())
    >>> # Shows what features to change to reduce interval width by 50%

    Notes
    -----
    Counterfactual explanations identify actionable interventions to reduce
    prediction uncertainty. For example:
    - "If we measure temperature more precisely, demand forecast uncertainty
       would decrease by 40%"
    - "Adding a promotion flag feature would halve our inventory uncertainty"

    The search minimizes both:
    1. Prediction interval width (uncertainty reduction)
    2. Feature perturbation magnitude (minimal change principle)
    """

    def __init__(
        self,
        model: "BaseUncertaintyModel",
        confidence: float = 0.9,
        method: str = "auto",
        random_state: int | None = None,
    ):
        self.model = model
        self.confidence = confidence
        self.method = method
        self.random_state = random_state

        # Initialize searcher based on method
        self._searcher = self._init_searcher()

    def _init_searcher(self) -> EvolutionarySearch | GradientSearch:
        """Initialize appropriate search strategy."""
        if self.method == "auto":
            # Auto-detect based on model type
            if self._is_differentiable_model():
                return GradientSearch(
                    self.model,
                    confidence=self.confidence,
                    random_state=self.random_state,
                )
            else:
                return EvolutionarySearch(
                    self.model,
                    confidence=self.confidence,
                    random_state=self.random_state,
                )
        elif self.method == "gradient":
            return GradientSearch(
                self.model,
                confidence=self.confidence,
                random_state=self.random_state,
            )
        elif self.method == "evolutionary":
            return EvolutionarySearch(
                self.model,
                confidence=self.confidence,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Invalid method: {self.method}. Must be 'auto', 'evolutionary', or 'gradient'"
            )

    def _is_differentiable_model(self) -> bool:
        """Check if model is differentiable (neural network)."""
        # Check for PyTorch models
        model_name = self.model.__class__.__name__
        if "Torch" in model_name or "Deep" in model_name:
            return True

        # Check for model attribute indicating differentiability
        if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
            return True

        return False

    def explain_uncertainty(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
        **search_kwargs,
    ) -> SearchResult:
        """
        Find counterfactual that reduces prediction interval width.

        Searches for minimal feature changes that achieve the target reduction
        in prediction interval width.

        Args:
            data: Feature DataFrame (typically single row)
            target_reduction: Target proportional reduction in interval width (0-1)
                - 0.5 = reduce interval width by 50%
                - 0.1 = reduce interval width by 10%
            feature_bounds: Optional bounds for each feature (min, max)
                - Ensures counterfactual values stay within realistic ranges
            fixed_features: Features that should not be modified
                - Useful when only certain features can be intervened upon
            **search_kwargs: Additional arguments passed to search strategy
                - For evolutionary: population_size, n_generations, mutation_rate, etc.
                - For gradient: learning_rate, n_iterations, l1_penalty, etc.

        Returns
        -------
        SearchResult
            Counterfactual explanation with:
            - counterfactual: Counterfactual feature values
            - original: Original feature values
            - changes: Per-feature changes (counterfactual - original)
            - interval_width_reduction: Achieved proportional reduction
            - original_width: Original interval width
            - new_width: Counterfactual interval width

        Raises
        ------
        InvalidDataError
            If data is empty or has more than one row

        Examples
        --------
        >>> # Find changes to halve interval width
        >>> result = explainer.explain_uncertainty(X_test.head(1), target_reduction=0.5)
        >>>
        >>> # Find changes with custom feature bounds
        >>> result = explainer.explain_uncertainty(
        ...     X_test.head(1),
        ...     target_reduction=0.3,
        ...     feature_bounds={"price": (0, 100), "promotion": (0, 1)}
        ... )
        >>>
        >>> # Find changes while keeping certain features fixed
        >>> result = explainer.explain_uncertainty(
        ...     X_test.head(1),
        ...     target_reduction=0.4,
        ...     fixed_features=["date", "category"]
        ... )

        Notes
        -----
        The search balances two objectives:
        1. Reducing prediction interval width (uncertainty reduction)
        2. Minimizing feature perturbations (minimal change principle)

        This multi-objective optimization is handled by combining:
        - Primary objective: Width reduction (achieve target)
        - Secondary objective: L1/L2 penalties on feature changes

        For tree-based models (evolutionary search), this uses a genetic
        algorithm with tournament selection, crossover, and mutation.

        For differentiable models (gradient search), this uses gradient
        descent with L1/L2 regularization.
        """
        from ..utils.exceptions import error_invalid_data

        if data.height == 0:
            error_invalid_data("Cannot explain uncertainty on empty DataFrame")

        if data.height > 1:
            error_invalid_data(
                "explain_uncertainty expects exactly one row. "
                "Use explain_batch() for multiple samples."
            )

        return self._searcher.search(
            data,
            target_reduction=target_reduction,
            feature_bounds=feature_bounds,
            fixed_features=fixed_features,
            **search_kwargs,
        )

    def explain_batch(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
        **search_kwargs,
    ) -> list[SearchResult]:
        """
        Generate counterfactual explanations for multiple samples.

        Args:
            data: Feature DataFrame with multiple rows
            target_reduction: Target proportional reduction in interval width
            feature_bounds: Optional bounds for each feature
            fixed_features: Features that should not be modified
            **search_kwargs: Additional arguments for search strategy

        Returns
        -------
        list[SearchResult]
            List of counterfactual explanations, one per input row

        Examples
        --------
        >>> results = explainer.explain_batch(X_test.head(10), target_reduction=0.4)
        >>> for i, result in enumerate(results):
        ...     print(f"Sample {i}: {result.interval_width_reduction:.1%} reduction")
        """
        results = []
        for i in range(data.height):
            row = data[i]
            result = self._searcher.search(
                row,
                target_reduction=target_reduction,
                feature_bounds=feature_bounds,
                fixed_features=fixed_features,
                **search_kwargs,
            )
            results.append(result)

        return results

    def compare_features(
        self,
        data: pl.DataFrame,
        features: list[str],
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> pl.DataFrame:
        """
        Compare impact of modifying individual features on uncertainty.

        For each feature, finds counterfactual with only that feature modifiable
        (all others fixed). This identifies which features are most effective
        at reducing uncertainty.

        Args:
            data: Feature DataFrame (single row)
            features: List of features to compare
            target_reduction: Target reduction for each feature search
            feature_bounds: Bounds for feature modifications

        Returns
        -------
        pl.DataFrame
            Comparison with columns:
                - feature: Feature name
                - width_reduction: Achieved proportional reduction
                - change_magnitude: Absolute change in feature value
                - effectiveness: Reduction per unit change

        Examples
        --------
        >>> comparison = explainer.compare_features(
        ...     X_test.head(1),
        ...     features=["temperature", "humidity", "pressure"],
        ...     target_reduction=0.3
        ... )
        >>> print(comparison.sort("effectiveness", descending=True))
        """
        from ..utils.exceptions import error_invalid_data

        if data.height != 1:
            error_invalid_data("compare_features requires single-row DataFrame")

        results = []

        for feature in features:
            # Fix all features except this one
            other_features = [f for f in data.columns if f != feature]

            try:
                result = self._searcher.search(
                    data,
                    target_reduction=target_reduction,
                    feature_bounds=feature_bounds,
                    fixed_features=other_features,
                )

                change_magnitude = abs(result.changes.get(feature, 0))
                effectiveness = (
                    result.interval_width_reduction / (change_magnitude + 1e-8)
                    if change_magnitude > 0
                    else 0
                )

                results.append(
                    {
                        "feature": feature,
                        "width_reduction": result.interval_width_reduction,
                        "change_magnitude": change_magnitude,
                        "effectiveness": effectiveness,
                    }
                )
            except Exception:
                # Feature search failed, skip
                results.append(
                    {
                        "feature": feature,
                        "width_reduction": 0.0,
                        "change_magnitude": 0.0,
                        "effectiveness": 0.0,
                    }
                )

        return pl.DataFrame(results).sort("effectiveness", descending=True)

    def summary(self) -> dict[str, Any]:
        """
        Return summary of the explainer configuration.

        Returns
        -------
        dict
            Configuration summary with keys:
                - confidence: Confidence level for intervals
                - method: Search strategy used
                - random_state: Random seed
                - model_type: Type of underlying model
        """
        return {
            "confidence": self.confidence,
            "method": self.method,
            "random_state": self.random_state,
            "model_type": self.model.__class__.__name__,
        }
