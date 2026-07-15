# ADR-005: Evidence Export and Documentation Site Boundary

**Status:** Accepted
**Date:** 2026-07-13

## Context

The repository currently publishes MkDocs documentation under the generated `site/` directory. The target architecture also calls for an Astro/Starlight evidence portal backed by compact generated run records.

## Decision

Keep MkDocs and the existing generated `site/` workflow unchanged during the pipeline migration. Build the evidence portal in a separate `evidence-site/` directory. Export only verified Platinum summaries as versioned, partitioned `jsonl.gz` records plus a small `index.json` containing partition paths, counts, schema versions, generation time, and checksums.

The Astro build will parse selected partitions at build time. Prediction-level Parquet artifacts remain in the artifact store and are not copied into the static site.

## Consequences

- The two documentation concerns have independent failure modes and review boundaries.
- Existing library documentation remains deployable while evidence publishing is developed.
- A later site cutover requires an explicit decision rather than silently replacing the current GitHub Pages output.
