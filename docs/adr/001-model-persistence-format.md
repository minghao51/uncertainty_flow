# ADR-001: Model Persistence Format

**Status:** Accepted
**Date:** 2026-04-26

## Context

We need a way to save and load fitted `BaseUncertaintyModel` subclasses to disk. The format must support:
- Model state (fitted parameters, hyperparameters)
- Metadata (quantile levels, target names, library version)
- Compatibility across Python/numpy versions

## Decision

We currently use **Python pickle** (`pickle.HIGHEST_PROTOCOL`) wrapped in a ZIP archive alongside JSON metadata. This was chosen for simplicity and to preserve complex sklearn model objects without custom serialization.

## Security Concern

Pickle can execute arbitrary code during deserialization. A malicious `.uf` file could compromise the system. The README already warns users to only load archives from trusted sources, but this is not ideal for a library.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| **cloudpickle + class whitelist** | Drop-in replacement, preserves model objects | Still executes code from potentially untrusted sources |
| **joblib** | sklearn's default, efficient for numpy arrays | Same pickle lineage, not fundamentally safer |
| **JSON + numpy arrays** | No code execution risk, portable, inspectable | Loses sklearn model objects — requires state extraction |
| **protobuf / msgpack** | Schema-based, no arbitrary code execution | Requires defining schemas, more work |

## Resolution for v0.2+

1. **Format v2**: Introduce a new archive format that saves model state as JSON + numpy arrays instead of pickle
2. **Backward compatibility**: v1 pickle format still supported on load with a deprecation warning
3. **Migration path**: Provide a utility to migrate v1 archives to v2 format

## Consequences

- Breaking change for any code that programmatically inspects `.uf` archives
- New save path (v2) will be safe by default
- Load path must handle both v1 (pickle) and v2 (JSON+numpy) formats
