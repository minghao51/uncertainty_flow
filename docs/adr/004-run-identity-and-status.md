# ADR-004: Deterministic Run Identity and Status Semantics

**Status:** Accepted
**Date:** 2026-07-13

## Context

`BenchmarkResult` currently uses a random UUID fragment. This prevents reliable reuse, idempotency, and cross-artifact lineage. Failure behavior also needs to distinguish required evaluation failures from optional diagnostic failures.

## Decision

Derive dataset, validation-plan, and run identities from canonical serialized inputs and SHA-256 hashes. Volatile timestamps are metadata only. A completed run is immutable. The coordinator applies an explicit `reuse_verified`, `fail_if_exists`, or `rerun` policy before acquiring a run lock.

Use these statuses:

```text
planned -> running -> success
                   -> degraded
                   -> failed
                   -> cancelled
```

Required branch failures block Platinum publication. Optional diagnostic failures produce `degraded` status and record the node, exception category, requested configuration, evidence impact, and remediation.

## Consequences

- Repeated identical requests can safely reuse verified artifacts.
- Run manifests become machine-verifiable operational records.
- Compatibility translation must map random legacy IDs separately from new content-derived IDs.
