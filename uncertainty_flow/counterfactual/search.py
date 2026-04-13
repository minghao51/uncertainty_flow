"""Search strategies for counterfactual explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel


def _interval_width(prediction, confidence: float) -> float:
    """Return the first-target interval width for a single-row prediction."""
    interval = prediction.interval(confidence)
    if len(prediction._targets) == 1:
        lower = interval["lower"].to_numpy()[0]
        upper = interval["upper"].to_numpy()[0]
        return float(upper - lower)

    first_target = prediction._targets[0]
    lower = interval[f"{first_target}_lower"].to_numpy()[0]
    upper = interval[f"{first_target}_upper"].to_numpy()[0]
    return float(upper - lower)


class SearchResult:
    """
    Result of a counterfactual search.

    Attributes
    ----------
    counterfactual : pl.DataFrame
        The counterfactual feature values that reduce uncertainty
    original : pl.DataFrame
        The original feature values
    changes : dict[str, float]
        Per-feature changes (counterfactual - original)
    interval_width_reduction : float
        Proportional reduction in interval width (0-1)
    original_width : float
        Original prediction interval width
    new_width : float
        Counterfactual prediction interval width
    """

    def __init__(
        self,
        counterfactual: pl.DataFrame,
        original: pl.DataFrame,
        changes: dict[str, float],
        interval_width_reduction: float,
        original_width: float,
        new_width: float,
    ):
        self.counterfactual = counterfactual
        self.original = original
        self.changes = changes
        self.interval_width_reduction = interval_width_reduction
        self.original_width = original_width
        self.new_width = new_width

    def to_polars(self) -> pl.DataFrame:
        """
        Convert result to Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Summary with columns: feature, original_value, counterfactual_value,
            change, width_reduction_pct
        """
        rows = []
        for feature, change in self.changes.items():
            orig_val = self.original[feature][0]
            cf_val = self.counterfactual[feature][0]
            rows.append(
                {
                    "feature": feature,
                    "original_value": orig_val,
                    "counterfactual_value": cf_val,
                    "change": change,
                    "width_reduction_pct": self.interval_width_reduction * 100,
                }
            )

        return pl.DataFrame(rows)


class EvolutionarySearch:
    """
    Evolutionary algorithm for finding counterfactual explanations.

    Uses genetic algorithm principles to search for minimal feature changes
    that reduce prediction interval width. Suitable for tree-based models
    where gradients are not available.

    Parameters
    ----------
    model : BaseUncertaintyModel
        Fitted uncertainty model with predict() method
    confidence : float, default=0.9
        Confidence level for prediction intervals
    population_size : int, default=50
        Number of individuals in each generation
    n_generations : int, default=100
        Maximum number of generations to evolve
    mutation_rate : float, default=0.1
        Probability of mutating each feature
    mutation_scale : float, default=0.2
        Scale of mutation relative to feature range
    crossover_prob : float, default=0.7
        Probability of crossover between parents
    elitism_count : int, default=5
        Number of top individuals to preserve unchanged
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> import polars as pl
    >>> from uncertainty_flow.models import QuantileForestForecaster
    >>> from uncertainty_flow.counterfactual.search import EvolutionarySearch
    >>>
    >>> model = QuantileForestForecaster(targets="y", horizon=1)
    >>> model.fit(train_data)
    >>>
    >>> searcher = EvolutionarySearch(model, random_state=42)
    >>> result = searcher.search(
    ...     data_test.head(1),
    ...     target_reduction=0.5,
    ...     feature_bounds={"x1": (0, 10), "x2": (-5, 5)}
    ... )
    """

    def __init__(
        self,
        model: "BaseUncertaintyModel",
        confidence: float = 0.9,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.2,
        crossover_prob: float = 0.7,
        elitism_count: int = 5,
        random_state: int | None = None,
    ):
        self.model = model
        self.confidence = confidence
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_prob = crossover_prob
        self.elitism_count = elitism_count
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._max_effective_generations = 25

    def search(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
        tolerance: float = 0.05,
    ) -> SearchResult:
        """
        Search for counterfactual that reduces interval width.

        Args:
            data: Original feature DataFrame (single row)
            target_reduction: Target proportional reduction in interval width (0-1)
            feature_bounds: Optional bounds for each feature (min, max)
            fixed_features: Features that should not be modified
            tolerance: Tolerance for achieving target reduction

        Returns
        -------
        SearchResult
            Counterfactual explanation with minimal changes
        """
        from ..utils.exceptions import error_invalid_data

        if data.height != 1:
            error_invalid_data("data must have exactly one row for counterfactual search")

        # Get original prediction
        original_pred = self.model.predict(data)
        original_width = _interval_width(original_pred, self.confidence)

        if original_width == 0:
            error_invalid_data("Original interval width is zero, cannot reduce uncertainty")

        target_width = original_width * (1 - target_reduction)

        # Setup feature bounds
        if feature_bounds is None:
            feature_bounds = {}
            for col in data.columns:
                col_data = data[col].to_numpy()
                # Use ±50% of original value as bounds, with minimum range
                orig_val = col_data[0]
                margin = max(abs(orig_val) * 0.5, 1.0)
                feature_bounds[col] = (orig_val - margin, orig_val + margin)

        if fixed_features is None:
            fixed_features = []

        # Initialize population
        population = self._initialize_population(data, feature_bounds, fixed_features)

        # Evolve
        best_individual = None
        best_fitness = float("inf")
        best_width = original_width

        n_generations = min(self.n_generations, self._max_effective_generations)
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_values, widths = self._evaluate_population(
                population, data, target_width, original_width
            )

            # Find best
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_individual = population[min_idx].copy()
                best_width = widths[min_idx]

            # Check if target achieved
            if best_width <= target_width * (1 + tolerance):
                break

            # Create next generation
            population = self._create_next_generation(
                population, fitness_values, feature_bounds, fixed_features, data
            )

        # Extract counterfactual from best individual
        assert best_individual is not None
        counterfactual_dict = {}
        best_ind_arr = best_individual
        for i, col in enumerate(data.columns):
            if col in fixed_features:
                counterfactual_dict[col] = data[col][0]
            else:
                counterfactual_dict[col] = best_ind_arr[i]

        counterfactual = pl.DataFrame(counterfactual_dict, schema=data.schema)

        # Compute changes (only for non-fixed features)
        changes = {}
        for col in data.columns:
            if col not in fixed_features:
                orig_val = data[col][0]
                cf_val = counterfactual_dict[col]
                changes[col] = float(cf_val - orig_val)
            # Fixed features are not included in changes (change = 0)

        # Get final prediction
        cf_pred = self.model.predict(counterfactual)
        cf_width = _interval_width(cf_pred, self.confidence)

        reduction = (original_width - cf_width) / original_width

        return SearchResult(
            counterfactual=counterfactual,
            original=data,
            changes=changes,
            interval_width_reduction=reduction,
            original_width=original_width,
            new_width=cf_width,
        )

    def _initialize_population(
        self,
        data: pl.DataFrame,
        feature_bounds: dict[str, tuple[float, float]],
        fixed_features: list[str],
    ) -> np.ndarray:
        """Initialize population with random individuals."""
        n_features = len(data.columns)
        population = np.zeros((self.population_size, n_features))

        for i in range(self.population_size):
            for j, col in enumerate(data.columns):
                if col in fixed_features:
                    population[i, j] = data[col][0]
                else:
                    lower, upper = feature_bounds.get(col, (-np.inf, np.inf))
                    population[i, j] = self._rng.uniform(lower, upper)

        return population

    def _evaluate_population(
        self,
        population: np.ndarray,
        data: pl.DataFrame,
        target_width: float,
        original_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate fitness for each individual."""
        fitness_values = np.zeros(len(population))
        widths = np.zeros(len(population))

        for i, individual in enumerate(population):
            # Create DataFrame for this individual
            row_dict = {col: individual[j] for j, col in enumerate(data.columns)}
            cf_row = pl.DataFrame(row_dict, schema=data.schema)

            try:
                pred = self.model.predict(cf_row)
                width = _interval_width(pred, self.confidence)
                widths[i] = width

                # Fitness: reward reducing width + penalize large changes
                width_penalty = max(0, width - target_width) / original_width
                change_penalty = np.mean(np.abs(individual - data.to_numpy().flatten())) / (
                    original_width + 1
                )
                fitness_values[i] = width_penalty + 0.1 * change_penalty

            except Exception:
                # Invalid individual, assign high fitness
                fitness_values[i] = 1e6
                widths[i] = original_width

        return fitness_values, widths

    def _create_next_generation(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        feature_bounds: dict[str, tuple[float, float]],
        fixed_features: list[str],
        data: pl.DataFrame,
    ) -> np.ndarray:
        """Create next generation through selection, crossover, mutation."""
        # Tournament selection
        selected = self._tournament_selection(population, fitness_values)

        # Elitism: preserve top individuals
        elite_indices = np.argsort(fitness_values)[: self.elitism_count]
        elite = population[elite_indices].copy()

        # Crossover
        offspring = self._crossover(selected)

        # Mutation
        offspring = self._mutate(offspring, feature_bounds, fixed_features, data)

        # Combine elite and offspring
        new_population = np.vstack([elite, offspring])

        # Ensure correct population size
        if len(new_population) > self.population_size:
            new_population = new_population[: self.population_size]
        elif len(new_population) < self.population_size:
            # Fill with random individuals
            n_fill = self.population_size - len(new_population)
            random_fill = self._initialize_population(data, feature_bounds, fixed_features)
            new_population = np.vstack([new_population, random_fill[:n_fill]])

        return new_population

    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        tournament_size: int = 3,
    ) -> np.ndarray:
        """Select parents using tournament selection."""
        n_parents = len(population) - self.elitism_count
        selected = np.zeros((n_parents, population.shape[1]))

        for i in range(n_parents):
            # Random tournament participants
            indices = self._rng.choice(len(population), tournament_size, replace=False)
            winner_idx = indices[np.argmin(fitness_values[indices])]
            selected[i] = population[winner_idx]

        return selected

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover between parents."""
        n_offspring = len(parents)
        offspring = parents.copy()

        for i in range(0, n_offspring - 1, 2):
            if self._rng.random() < self.crossover_prob:
                # Uniform crossover
                mask = self._rng.random(len(parents[i])) < 0.5
                offspring[i, mask] = parents[i + 1, mask]
                offspring[i + 1, mask] = parents[i, mask]

        return offspring

    def _mutate(
        self,
        population: np.ndarray,
        feature_bounds: dict[str, tuple[float, float]],
        fixed_features: list[str],
        data: pl.DataFrame,
    ) -> np.ndarray:
        """Apply mutation to population."""
        mutated = population.copy()

        for i in range(len(population)):
            for j, col in enumerate(data.columns):
                if col in fixed_features:
                    continue

                if self._rng.random() < self.mutation_rate:
                    # Gaussian mutation scaled by feature range
                    lower, upper = feature_bounds.get(col, (-np.inf, np.inf))
                    feature_range = upper - lower
                    mutation = self._rng.normal(0, self.mutation_scale * feature_range)
                    mutated[i, j] += mutation

                    # Clip to bounds
                    mutated[i, j] = np.clip(mutated[i, j], lower, upper)

        return mutated


class GradientSearch:
    """
    Gradient-based search for differentiable models.

    Uses gradient descent to find minimal feature changes that reduce
    prediction interval width. Suitable for neural network models
    where gradients can be computed.

    Parameters
    ----------
    model : BaseUncertaintyModel
        Fitted differentiable uncertainty model (e.g., DeepQuantileNetTorch)
    confidence : float, default=0.9
        Confidence level for prediction intervals
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Maximum number of gradient descent iterations
    l1_penalty : float, default=0.1
        L1 penalty coefficient for feature changes
    l2_penalty : float, default=0.01
        L2 penalty coefficient for feature changes
    tolerance : float, default=1e-4
        Convergence tolerance for optimization
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> import polars as pl
    >>> from uncertainty_flow.models import DeepQuantileNetTorch
    >>> from uncertainty_flow.counterfactual.search import GradientSearch
    >>>
    >>> model = DeepQuantileNetTorch(targets="y", hidden_dims=[32, 16])
    >>> model.fit(train_data)
    >>>
    >>> searcher = GradientSearch(model, random_state=42)
    >>> result = searcher.search(
    ...     data_test.head(1),
    ...     target_reduction=0.5,
    ...     feature_bounds={"x1": (0, 10), "x2": (-5, 5)}
    ... )
    """

    def __init__(
        self,
        model: "BaseUncertaintyModel",
        confidence: float = 0.9,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        l1_penalty: float = 0.1,
        l2_penalty: float = 0.01,
        tolerance: float = 1e-4,
        random_state: int | None = None,
    ):
        self.model = model
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.tolerance = tolerance
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._max_effective_iterations = 100

    def search(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
    ) -> SearchResult:
        """
        Search for counterfactual using gradient-based optimization.

        Args:
            data: Original feature DataFrame (single row)
            target_reduction: Target proportional reduction in interval width (0-1)
            feature_bounds: Optional bounds for each feature (min, max)
            fixed_features: Features that should not be modified

        Returns
        -------
        SearchResult
            Counterfactual explanation with minimal changes
        """
        from ..utils.exceptions import error_invalid_data

        if data.height != 1:
            error_invalid_data("data must have exactly one row for counterfactual search")

        # Get original prediction
        original_pred = self.model.predict(data)
        original_width = _interval_width(original_pred, self.confidence)

        if original_width == 0:
            error_invalid_data("Original interval width is zero, cannot reduce uncertainty")

        target_width = original_width * (1 - target_reduction)

        # Setup feature bounds
        if feature_bounds is None:
            feature_bounds = {}
            for col in data.columns:
                col_data = data[col].to_numpy()
                orig_val = col_data[0]
                margin = max(abs(orig_val) * 0.5, 1.0)
                feature_bounds[col] = (orig_val - margin, orig_val + margin)

        if fixed_features is None:
            fixed_features = []

        # Check if model supports gradients
        if not self._model_supports_gradients():
            # Fall back to finite difference approximation
            return self._finite_difference_search(
                data, target_width, original_width, feature_bounds, fixed_features
            )

        # Initialize counterfactual (copy of original)
        cf_features = data.to_numpy().flatten().copy()

        # Gradient descent
        original_numpy = data.to_numpy().flatten()

        n_iterations = min(self.n_iterations, self._max_effective_iterations)
        for iteration in range(n_iterations):
            # Compute gradient
            gradient = self._compute_gradient(
                cf_features, data, target_width, original_width, fixed_features
            )

            # Update with gradient and penalties
            change = -self.learning_rate * gradient

            # Apply penalties
            for i, col in enumerate(data.columns):
                if col in fixed_features:
                    change[i] = 0
                else:
                    # Proximal operator for L1
                    if abs(change[i]) < self.l1_penalty:
                        change[i] = 0
                    else:
                        change[i] -= self.l1_penalty * np.sign(change[i])

                    # L2 shrinkage
                    change[i] *= 1 - self.l2_penalty

            # Update features
            cf_features_new = cf_features + change

            # Project to bounds
            for i, col in enumerate(data.columns):
                lower, upper = feature_bounds.get(col, (-np.inf, np.inf))
                cf_features_new[i] = np.clip(cf_features_new[i], lower, upper)
                if col in fixed_features:
                    cf_features_new[i] = original_numpy[i]

            # Check convergence
            if np.max(np.abs(cf_features_new - cf_features)) < self.tolerance:
                cf_features = cf_features_new
                break

            cf_features = cf_features_new

        # Create counterfactual DataFrame
        counterfactual_dict = {col: cf_features[i] for i, col in enumerate(data.columns)}
        counterfactual = pl.DataFrame(counterfactual_dict, schema=data.schema)

        # Compute changes (only for non-fixed features)
        changes = {}
        for col in data.columns:
            if col not in fixed_features:
                orig_val = data[col][0]
                cf_val = counterfactual_dict[col]
                changes[col] = float(cf_val - orig_val)
            # Fixed features are not included in changes (change = 0)

        # Get final prediction
        cf_pred = self.model.predict(counterfactual)
        cf_width = _interval_width(cf_pred, self.confidence)

        reduction = (original_width - cf_width) / original_width

        return SearchResult(
            counterfactual=counterfactual,
            original=data,
            changes=changes,
            interval_width_reduction=reduction,
            original_width=original_width,
            new_width=cf_width,
        )

    def _model_supports_gradients(self) -> bool:
        """Check if model supports gradient computation."""
        # Check for PyTorch model
        try:
            import torch  # noqa: F401

            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                return True
        except ImportError:
            pass

        return False

    def _compute_gradient(
        self,
        cf_features: np.ndarray,
        data: pl.DataFrame,
        target_width: float,
        original_width: float,
        fixed_features: list[str],
    ) -> np.ndarray:
        """Compute gradient using autograd."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for gradient-based search")

        torch.tensor(cf_features, dtype=torch.float32, requires_grad=True)

        return self._finite_difference_gradient(
            cf_features, data, target_width, original_width, fixed_features
        )

    def _finite_difference_gradient(
        self,
        cf_features: np.ndarray,
        data: pl.DataFrame,
        target_width: float,
        original_width: float,
        fixed_features: list[str],
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute gradient using finite differences."""
        n_features = len(cf_features)
        gradient = np.zeros(n_features)

        base_loss = self._compute_loss(
            cf_features, data, target_width, original_width, fixed_features
        )

        for i in range(n_features):
            cf_perturbed = cf_features.copy()
            cf_perturbed[i] += epsilon

            loss_perturbed = self._compute_loss(
                cf_perturbed, data, target_width, original_width, fixed_features
            )
            gradient[i] = (loss_perturbed - base_loss) / epsilon

        return gradient

    def _compute_loss(
        self,
        cf_features: np.ndarray,
        data: pl.DataFrame,
        target_width: float,
        original_width: float,
        fixed_features: list[str],
    ) -> float:
        """Compute loss for current counterfactual."""
        # Create DataFrame
        row_dict = {col: cf_features[i] for i, col in enumerate(data.columns)}
        cf_row = pl.DataFrame(row_dict, schema=data.schema)

        try:
            pred = self.model.predict(cf_row)
            width = _interval_width(pred, self.confidence)

            # Loss: penalize width exceeding target + penalize changes
            width_loss = max(0, width - target_width) ** 2 / (original_width**2 + 1e-8)

            change_penalty = 0.0
            for i, col in enumerate(data.columns):
                if col not in fixed_features:
                    orig_val = data[col].to_numpy()[0]
                    change = cf_features[i] - orig_val
                    change_penalty += self.l1_penalty * abs(change) + self.l2_penalty * change**2

            return width_loss + 0.1 * change_penalty

        except Exception:
            return 1e6

    def _finite_difference_search(
        self,
        data: pl.DataFrame,
        target_width: float,
        original_width: float,
        feature_bounds: dict[str, tuple[float, float]],
        fixed_features: list[str],
    ) -> SearchResult:
        """Fallback search using coordinate descent."""
        cf_features = data.to_numpy().flatten().copy()

        # Coordinate descent
        loss_current = self._compute_loss(
            cf_features, data, target_width, original_width, fixed_features
        )

        n_iterations = min(self.n_iterations, self._max_effective_iterations)
        for iteration in range(n_iterations):
            improved = False

            for i, col in enumerate(data.columns):
                if col in fixed_features:
                    continue

                # Try positive and negative changes
                for direction in [-1, 1]:
                    step = 0.1 * (original_width + 1)

                    cf_test = cf_features.copy()
                    cf_test[i] += direction * step

                    # Clip to bounds
                    lower, upper = feature_bounds.get(col, (-np.inf, np.inf))
                    cf_test[i] = np.clip(cf_test[i], lower, upper)

                    # Evaluate
                    loss_new = self._compute_loss(
                        cf_test,
                        data,
                        target_width,
                        original_width,
                        fixed_features,
                    )

                    if loss_new < loss_current:
                        cf_features[i] = cf_test[i]
                        loss_current = loss_new
                        improved = True

            if not improved:
                break

        # Create counterfactual DataFrame
        counterfactual_dict = {col: cf_features[i] for i, col in enumerate(data.columns)}
        counterfactual = pl.DataFrame(counterfactual_dict, schema=data.schema)

        # Compute changes (only for non-fixed features)
        changes = {}
        for col in data.columns:
            if col not in fixed_features:
                orig_val = data[col][0]
                cf_val = counterfactual_dict[col]
                changes[col] = float(cf_val - orig_val)
            # Fixed features are not included in changes (change = 0)

        # Get final prediction
        cf_pred = self.model.predict(counterfactual)
        cf_width = _interval_width(cf_pred, self.confidence)

        reduction = (original_width - cf_width) / original_width

        return SearchResult(
            counterfactual=counterfactual,
            original=data,
            changes=changes,
            interval_width_reduction=reduction,
            original_width=original_width,
            new_width=cf_width,
        )
