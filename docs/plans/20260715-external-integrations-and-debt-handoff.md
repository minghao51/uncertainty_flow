# External Integrations and Repository Debt Handoff

**Status:** Ready for implementation after deployment decisions
**Prepared:** 2026-07-15
**Predecessor:** `docs/plans/20260713-hamilton-medallion-refactor-plan.md`

## Objective

Complete the production-facing integrations around the verified local Hamilton/medallion benchmark pipeline, then close the remaining bounded repository debt without weakening the local publication, identity, verification, or clean-cutover contracts.

This handoff separates work that requires an explicit platform decision from work that can be completed locally. Do not select a cloud, scheduler, alert provider, or key-management service implicitly.

## Current verified baseline

- The registry-backed pipeline is authoritative; legacy benchmark orchestration and compatibility modules are removed.
- Local execution stages Bronze/Silver/Gold/Platinum artifacts, verifies checksums, publishes the manifest last, and supports verified reuse.
- The CLI uses lock-aware coordinators and exposes plan, run, verify, lineage, list-runs, export-site, and conservative GC commands.
- `uncertainty_flow/benchmarking/deployment.py` contains provider-neutral `ObjectStore`, `Scheduler`, and `AlertSink` protocols plus safe local reference implementations. These protocols are not yet injected into coordinator execution or publication.
- Local locking is filesystem-scoped. It is insufficient when multiple workers publish to shared object storage.
- Manifest HMAC support exists, but secrets are supplied directly in request publication settings and there is no external signing service.
- `RetentionPolicy` only plans deletion of unverified local runs. Verified-history expiry is reported but deliberately not executed.
- The evidence site builds locally from generated verified summaries, but no production hosting/cutover decision has been made.
- Last completed release gate: `1110 passed, 45 skipped`, 85.62% coverage, Ruff and package mypy passing.

Re-run the baseline before implementation; the counts above are a dated snapshot, not a permanent assertion.

## Invariants

Every implementation package must preserve these rules:

1. A run identity is derived from normalized source, dataset, validation, model, evaluation, and code-version inputs; credentials and storage location do not alter scientific identity.
2. Credentials, signing keys, access tokens, and provider secrets never appear in resolved configuration, manifests, logs, scheduler payloads, evidence exports, or exception text.
3. Required failures block publication. Only explicitly optional branches may publish a degraded run.
4. A final manifest is visible only after every referenced artifact and verification report is durably written and verified.
5. Reuse requires matching identity, successful verification, valid checksums, and the configured authenticity policy.
6. Dry-run and plan operations perform no external writes, scheduler submissions, alerts, or lock acquisition.
7. GC and retention default to preview mode and never delete verified history without a separate explicit policy and operator acknowledgement.
8. Provider SDK types remain behind deployment adapters; Hamilton nodes and typed benchmark contracts stay provider-neutral.

## Decisions required before external implementation

Record the selected values in `docs/operations/deployment.md` and an ADR before starting E1:

| Decision | Required choice | Acceptance constraint |
| --- | --- | --- |
| Artifact store | S3, GCS, Azure Blob, or another concrete store | Must support immutable/versioned object writes, checksum metadata, and manifest-last publication. |
| Execution platform | Airflow, Kubernetes Jobs, GitHub Actions, or another scheduler | Submission must be idempotent by run identity and return a durable external execution ID. |
| Distributed lock | Database lease, object-store conditional write, Redis, or scheduler-native concurrency | Must provide owner tokens, expiry/renewal, safe release, and contention tests. |
| Signing | AWS KMS, GCP KMS, Azure Key Vault, Vault, or another managed signer | Private key material must not enter the process; verification metadata must identify key/version/algorithm. |
| Alerts | Slack, PagerDuty, email, or internal incident service | Alert delivery must be retryable and deduplicated by run/status transition. |
| Retention | Regulatory/business retention periods per artifact class | Verified-run deletion requires explicit approval and an auditable tombstone. |
| Evidence hosting | Existing GitHub Pages, separate static hosting, or no public deployment | Generated evidence must contain summaries only and must not expose prediction-level or secret data. |

## Work packages

### E0 — freeze the deployment contract

**Purpose:** convert the selected platform into explicit capabilities before adding SDK code.

1. Add an ADR naming the selected services, credential model, tenancy boundaries, regions, encryption requirements, and failure/retry expectations.
2. Expand `ObjectStore` into the smallest contract needed by publication: immutable put, streaming read, existence/head metadata, checksum/ETag, conditional create, list-by-prefix for recovery, and explicit deletion.
3. Add provider-neutral distributed lock and signing protocols. Do not return `RunLockManager` from a function named `distributed_lock_manager` once shared storage is supported.
4. Define typed deployment configuration with secret references rather than secret values.
5. Specify transient versus terminal adapter errors and retry ownership.

**Files:**

- `uncertainty_flow/benchmarking/deployment.py`
- `uncertainty_flow/benchmarking/contracts/`
- `docs/operations/deployment.md`
- new `docs/adr/006-*.md`
- `tests/benchmarking/test_deployment.py`

**Gate:** contract tests cover conditional publication, checksum metadata, lock ownership/expiry, signer metadata, and secret-safe serialization without importing a provider SDK.

### E1 — implement the selected artifact store and publication transaction

1. Implement the concrete object-store adapter in a dedicated optional module.
2. Introduce a storage factory selected by typed deployment configuration.
3. Preserve local filesystem behavior and test both backends against one shared artifact-store conformance suite.
4. For object storage, publish immutable run objects under a staging prefix, verify each object, then conditionally create the final manifest/catalog pointer last.
5. Define rerun behavior without relying on filesystem rename or directory replacement.
6. Add recovery for abandoned staging prefixes and partial uploads.

**Files:**

- `uncertainty_flow/benchmarking/storage/base.py`
- `uncertainty_flow/benchmarking/storage/local.py`
- new `uncertainty_flow/benchmarking/storage/<provider>.py`
- `uncertainty_flow/benchmarking/coordinator.py`
- `uncertainty_flow/benchmarking/matrix.py`
- shared storage conformance tests under `tests/benchmarking/`

**Gate:** fault-injection tests prove that no failed or partial upload becomes a reusable published run and that a manifest never references a missing/corrupt object.

### E2 — distributed locking and scheduler integration

1. Implement lease acquisition, renewal/heartbeat, owner-token release, stale recovery, and contention behavior.
2. Inject the lock implementation into coordinators; retain local `RunLockManager` as the default filesystem adapter.
3. Implement scheduler submission keyed by run identity. Repeated submissions must return/reconcile the same active execution rather than launch duplicates.
4. Store only secret references in scheduler payloads. Resolve credentials in the worker environment.
5. Define cancellation semantics and the final run status for user cancellation versus infrastructure loss.

**Gate:** concurrent integration tests show exactly one publisher for a run identity, safe lease loss behavior, idempotent scheduler submission, and no orphaned successful manifest after cancellation.

### E3 — managed signing, alerts, and operational events

1. Replace direct request-secret signing in production configuration with a signer protocol backed by the selected managed service.
2. Extend manifest authenticity metadata with algorithm, key identifier/version, and signature while preserving canonical payload rules.
3. Verify signatures during reuse, CLI verification, and evidence export according to the configured trust policy.
4. Emit structured lifecycle events for submission, start, node failure, degraded publication, verification failure, success, cancellation, and retention action.
5. Route alerts from status transitions through a deduplicating, retryable adapter. Alert failure must not rewrite scientific run status.

**Gate:** tests cover key rotation, unknown/disabled keys, tampering, signer outage, alert retries/deduplication, and proof that secret material never enters artifacts or logs.

### E4 — retention, recovery, and evidence deployment

1. Define separate retention classes for abandoned staging, failed/unverified runs, verified manifests/metrics, predictions, fitted models, and compact evidence exports.
2. Keep preview as the default. Require an explicit apply flag and policy identifier for destructive actions.
3. Add auditable tombstones before any approved verified-history deletion.
4. Define restore/rebuild procedures from immutable artifacts and test evidence-index regeneration.
5. Implement the selected evidence hosting workflow, including a generated-evidence freshness check and static build verification.
6. Decide whether MkDocs and the evidence site remain separate or share one hosting surface; document the cutover rather than replacing GitHub Pages implicitly.

**Gate:** retention dry-run/apply tests, active-run protection, restore exercise, checksum verification, evidence build, and hosting smoke test pass.

### D1 — enforce a hard tuning timeout

The current timeout is cooperative: it prevents a new trial from starting after the deadline but cannot interrupt one long-running fit.

1. Execute each trial in a cancellable process boundary or a scheduler job with a deadline.
2. Terminate and reap timed-out workers without leaking processes or temporary artifacts.
3. Record timed-out trials distinctly from model/configuration failures.
4. Preserve deterministic candidate order and best-completed-trial selection.

**Files:** `uncertainty_flow/benchmarking/tuning.py`, tuning contracts/tests, CLI documentation.

**Gate:** a deliberately blocking trial is terminated within a bounded grace period and leaves no child process or published tuning result.

### D2 — make lock ownership unambiguous in the public coordinator API

The CLI uses `run_with_lock()`, while public `run()` remains an unlocked lower-level path and rolling delegation relies on that distinction to avoid re-entrant deadlock.

1. Choose one explicit API contract: make public execution lock-aware and move unlocked execution to a private method, or rename the unlocked method so callers cannot mistake it for the safe default.
2. Refactor rolling-origin delegation without nested acquisition of the same run lock.
3. Add concurrent direct-library tests and update guides/API docs.

**Gate:** every documented/public execution path is lock-safe, and internal delegation acquires exactly one run lock.

### D3 — close documentation and formatting debt

1. Format `scripts/report_touched_modules.py` and add it to the normal formatting gate.
2. Fix the existing Griffe warnings by adding accurate public parameter annotations/docstrings; do not suppress them globally.
3. Repair dated archive links in `docs/archive/README.md` and the changelog link that escapes the docs tree.
4. Add the accepted ADRs, operations pages, and active handoff plans to MkDocs navigation, or explicitly exclude archival material.
5. Remove stale compatibility-era wording from ADR-003 and ADR-004 now that the clean cutover is complete.
6. Change documentation CI to `mkdocs build --strict` only after the warning set is zero.

**Gate:** `uv run ruff format . --check` and `uv run mkdocs build --strict` both pass.

### D4 — strengthen CI for the new supported surfaces

1. Add the evidence-site install/build command to CI with dependency caching and a lockfile-based install.
2. Add a test that generated evidence files remain ignored while `public/evidence/.gitkeep` remains trackable.
3. Add object-store conformance and deployment adapter tests to the appropriate optional/integration lane.
4. Preserve the full coverage-bearing suite as the release gate; benchmark-only `--no-cov` tests remain a fast development lane.
5. Add isolated base-install CLI smoke coverage so optional YAML/Hamilton/provider imports do not regress command help.

**Gate:** CI reproduces Ruff, full formatting, mypy, full coverage, strict docs, evidence-site build, and deployment conformance checks from a clean checkout.

### D5 — remaining contract hardening

1. Replace the static package code-version default with an injected immutable build/source revision in deployment while retaining a deterministic local fallback.
2. Validate remote dataset revision syntax/trust policy before network loading.
3. Expand provider applicability and tuning schemas so unsupported dataset/model/validation combinations fail during planning.
4. Decide whether deeply immutable request mappings are required; if so, replace mutable nested dictionaries with typed frozen models at the contract boundary.

**Gate:** identity changes when source revision changes, untrusted/unpinned remote revisions fail before I/O, and unsupported provider combinations fail before staging.

## Recommended sequence

1. Complete D3 and D4 first where they do not depend on an external platform; this gives the later adapters strict CI coverage.
2. Record deployment decisions and complete E0.
3. Implement E1 and E2 together behind conformance tests because object-store publication and distributed locking share atomicity requirements.
4. Implement E3 after publication and lock ownership are stable.
5. Implement E4 after real artifact classes and hosting targets are known.
6. Complete D1, D2, and D5 as separate, reviewable correctness changes.

## Suggested change boundaries

1. **PR 1 — local debt gates:** D3 plus the non-provider portions of D4.
2. **PR 2 — deployment ADR and protocols:** selected decisions plus E0.
3. **PR 3 — object-store publication:** E1 and conformance tests.
4. **PR 4 — leases and scheduler:** E2.
5. **PR 5 — signer, events, and alerts:** E3.
6. **PR 6 — retention and evidence hosting:** E4.
7. **PR 7 — tuning/API/contract hardening:** D1, D2, and D5, split further if review size warrants it.

Do not combine provider SDK introduction, coordinator publication changes, and unrelated documentation cleanup in one change.

## Verification commands

Fast touched-slice checks:

```bash
uv run pytest -q --no-cov tests/benchmarking
uv run ruff check uncertainty_flow/benchmarking tests/benchmarking uncertainty_flow/cli.py
uv run ruff format uncertainty_flow/benchmarking tests/benchmarking uncertainty_flow/cli.py --check
uv run mypy uncertainty_flow/
git diff --check
```

Required release gate:

```bash
uv run ruff check .
uv run ruff format . --check
uv run mypy uncertainty_flow/
uv run pytest -q
uv run mkdocs build --strict
npm ci --prefix evidence-site
npm run build --prefix evidence-site
git diff --check
```

Add provider-emulator or sandbox integration commands to this section once E0 selects the target services. Never run destructive retention tests against a shared production account.

## Resume point

The next agent should begin with D3/D4 or pause for the E0 platform decisions. Before editing deployment code:

1. Read this handoff, ADR-003 through ADR-005, and `docs/operations/deployment.md`.
2. Confirm the chosen artifact store, scheduler, lock service, signer, alert sink, retention policy, and evidence host with the owner.
3. Re-run the current release gate and distinguish pre-existing debt from new failures.
4. Update this file after each work package with status, exact verification results, and any changed decision.
