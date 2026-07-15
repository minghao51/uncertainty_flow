# Pipeline Verification and Release Signatures

Every published run contains a verification report and a manifest. The manifest records the run identity, status, resolved configuration hash, artifact lineage, and verification result.

For release workflows that need authenticity in addition to checksum-based corruption detection, use the HMAC helpers in `uncertainty_flow.benchmarking.operations` with a secret managed outside the repository:

```python
from uncertainty_flow.benchmarking.operations import sign_manifest, verify_manifest_signature

signature = sign_manifest(manifest, release_secret)
assert verify_manifest_signature(manifest, signature, release_secret)
```

The secret must never be stored with the generated artifacts. HMAC proves possession of the release secret; teams requiring non-repudiation should replace this with an externally managed signing service.

Node events can be written as compressed JSONL using `NodeEventWriter`. Events are operational diagnostics and are not substitutes for the immutable run manifest or verification report.
