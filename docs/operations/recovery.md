# Pipeline Recovery

Phase 6 local operations are deliberately conservative.

## Run locks

Pipeline runs acquire a lock under `<storage-root>/_locks/<run-id>.lock`. An active lock blocks a second execution. Locks older than the configured stale threshold are recovered automatically before a new lock is created.

The CLI uses the lock-aware path for single-model and matrix runs:

```bash
uncertainty-flow pipeline run --config configs/benchmark.yaml
```

## Cleanup

Preview cleanup first:

```bash
uncertainty-flow pipeline gc --root data
```

Apply cleanup only after reviewing the result:

```bash
uncertainty-flow pipeline gc --root data --apply
```

Cleanup removes only failed, malformed, or incomplete run directories. Verified `success` and `degraded` runs are never removed by this command.

## Evidence recovery

Regenerate the compact evidence catalog from verified Platinum artifacts:

```bash
uncertainty-flow pipeline export-site --root data --output evidence-site/public/evidence
```

The exporter validates manifests, verification reports, metrics, record counts, gzip output, and partition checksums before writing the index.
