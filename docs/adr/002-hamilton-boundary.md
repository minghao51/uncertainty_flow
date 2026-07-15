# ADR-002: Hamilton Boundary for Benchmarking

**Status:** Accepted
**Date:** 2026-07-13

## Context

The current benchmark lifecycle is concentrated in `BenchmarkFlow`. The refactor needs explicit dependencies and selective execution without coupling the core model API to orchestration.

## Decision

Use Hamilton only inside `uncertainty_flow/benchmarking/`. Hamilton nodes will declare data dependencies and domain transformations. The run coordinator will own operational policy, configuration resolution, identity, retries, and final status. Core models, wrappers, metrics, and persistence remain usable without Hamilton.

Begin with a single vertical slice. Add reusable sub-DAGs only after contracts and node names are stable.

## Consequences

- The existing `BenchmarkFlow` remains a compatibility facade during migration.
- Nodes must be testable as ordinary Python functions.
- Side effects are isolated behind storage/materialization adapters.
- The Hamilton dependency is not introduced until Phase 2, after baseline compatibility is frozen.
