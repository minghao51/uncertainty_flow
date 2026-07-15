# Deployment Integrations

The repository now exposes provider-neutral boundaries for the remaining deployment concerns:

- `ObjectStore` for S3/GCS/Azure or filesystem artifact backends.
- `Scheduler` for Airflow, Kubernetes Jobs, CI runners, or another execution platform.
- `AlertSink` for Slack, email, PagerDuty, or an internal incident system.
- `RunLockManager` for filesystem deployments; replace it with a distributed implementation when multiple workers share object storage.
- `RetentionPolicy` and `plan_retention()` for operator-reviewed cleanup.

The checked-in reference adapters are intentionally safe and local:

- `LocalObjectStore` copies objects under a traversal-safe root.
- `RecordingScheduler` records requests without executing arbitrary work.
- `LoggingAlertSink` emits structured local/CI logs.

Concrete cloud or scheduler adapters should be added only after selecting the target platform and credential model. The coordinator should depend on these protocols, not on a provider SDK.
