from uncertainty_flow.benchmarking.configs import BenchmarkConfig, ModelBuildConfig, RunConfig


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig(dataset_name="weather")
        assert cfg.dataset_name == "weather"
        assert cfg.n_samples == 1000
        assert cfg.random_state == 42
        assert cfg.auto_tune is True
        assert cfg.target_coverage == 0.9
        assert cfg.test_size == 0.2
        assert cfg.confidence_levels == [0.8, 0.9, 0.95]

    def test_custom_values(self):
        cfg = RunConfig(
            dataset_name="exchange_rate",
            n_samples=500,
            random_state=7,
            auto_tune=False,
            target_coverage=0.8,
            test_size=0.3,
        )
        assert cfg.n_samples == 500
        assert cfg.random_state == 7
        assert cfg.auto_tune is False
        assert cfg.target_coverage == 0.8
        assert cfg.test_size == 0.3

    def test_confidence_levels_override(self):
        cfg = RunConfig(dataset_name="weather", confidence_levels=[0.5, 0.95])
        assert cfg.confidence_levels == [0.5, 0.95]


class TestModelBuildConfig:
    def test_defaults(self):
        cfg = ModelBuildConfig(model_name="qf", target_column="y")
        assert cfg.model_name == "qf"
        assert cfg.target_column == "y"
        assert cfg.horizon == 3
        assert cfg.n_estimators == 30
        assert cfg.random_state == 42
        assert cfg.tuned_params is None

    def test_custom(self):
        cfg = ModelBuildConfig(
            model_name="cr",
            target_column="price",
            horizon=5,
            n_estimators=50,
            tuned_params={"n_estimators": 20},
        )
        assert cfg.horizon == 5
        assert cfg.tuned_params == {"n_estimators": 20}


class TestBenchmarkConfig:
    def test_inherits_run_config(self):
        cfg = BenchmarkConfig(dataset_name="weather")
        assert cfg.dataset_name == "weather"
        assert isinstance(cfg, RunConfig)

    def test_extra_fields(self):
        cfg = BenchmarkConfig(
            dataset_name="weather",
            horizon=7,
            n_estimators=50,
            target_column="OT",
        )
        assert cfg.horizon == 7
        assert cfg.n_estimators == 50
        assert cfg.target_column == "OT"

    def test_post_init_confidence_levels(self):
        cfg = BenchmarkConfig(dataset_name="weather")
        assert cfg.confidence_levels == [0.8, 0.9, 0.95]
