# ADR-003: Medallion Artifact Contracts and Local Storage

**Status:** Accepted
**Date:** 2026-07-13

## Context

Benchmark outputs currently consist of JSON/CSV sink files. They do not provide immutable source snapshots, persisted split membership, checksums, lineage, or a verified publication boundary.

## Decision

Use four versioned artifact layers:

- Bronze: immutable source-faithful snapshot plus source schema/row metadata.
- Silver: normalized typed observations with a stable string row identity and normalization report.
- Gold: experiment-ready observations with split/fold membership and validation contract.
- Platinum: verified run outputs, metrics, predictions, diagnostics, lineage, model card, and final manifest.

The first implementation is a local filesystem `ArtifactStore` with atomic temporary-file-to-final-path publication. Every manifest records schema version, content checksum, row counts where applicable, and the code/configuration inputs used to produce it.

## Consequences

- Failed staging runs cannot appear in the published run catalog.
- Storage implementations can later be replaced by object-store adapters without changing Hamilton nodes.
- Existing JSON/CSV sinks remain compatibility outputs until parity is proven.
