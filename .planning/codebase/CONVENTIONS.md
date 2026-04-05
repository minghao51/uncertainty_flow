# Codebase Conventions

## Code Style (Formatting, Line Length, Quotes)

- **Line Length**: Maximum 100 characters (enforced by ruff)
- **Quotes**: Use single quotes for strings (ruff E713/E714 violation otherwise)
- **Formatting**: Code is formatted using ruff for linting
- **Python Version**: Target Python 3.11+ (minimal supported version)

## Naming Conventions

### Files
- Use snake_case for Python files
- Module names should be descriptive and lowercase
- Test files: `test_*.py`
- No underscores in __init__.py files

### Classes
- Use PascalCase (CapWords)
- Abstract base classes typically use "Base" suffix
- Exception classes use "Error" suffix
- Keep class names concise but descriptive

### Functions and Methods
- Use snake_case
- Private methods/functions use underscore prefix
- Property methods use property decorator
- Dunder methods (__init__, __str__, etc.) allowed

### Variables
- Use snake_case for local variables
- Instance attributes: `self.attribute_name`
- Class attributes: `ClassName.CONSTANT_NAME`
- Parameters: descriptive names using snake_case

## Type Annotation Patterns

- Use type hints consistently across the codebase
- Forward references using string literals (`"ClassName"`)
- Use Union types with `|` syntax (Python 3.10+ style)
- Optional types use `Type | None`
- Collection types: `list[Type]`, `dict[str, Type]`
- Polars types: `pl.DataFrame`, `pl.LazyFrame`, `pl.Series`
- Custom type aliases defined in `core/types.py`

```python
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..core.distribution import DistributionPrediction

def function(param: int | None) -> str | list[float]:
    pass
```

## Error Handling Patterns

- Custom exception hierarchy under `UncertaintyFlowError`
- All exceptions inherit from `ValueError` for backward compatibility
- Error codes follow pattern: `UF-E###` (errors) or `UF-W###` (warnings)
- Use helper functions for raising specific errors
- Contextual error messages with clear descriptions
- Include error codes in messages when available

```python
class DataError(UncertaintyFlowError):
    """Base class for data-related errors."""
    pass

def error_invalid_data(reason: str) -> None:
    """Raise InvalidDataError."""
    raise InvalidDataError(reason)
```

## Logging Patterns

- **No logging**: The codebase does not use the standard logging module
- **Warnings**: Use Python's warnings module for user notifications
- **Warning categories**: Custom `UncertaintyFlowWarning` class
- **Stack level**: Use `stacklevel=3` to point to caller
- **Warning codes**: Include warning codes like `UF-W001`

```python
import warnings

def warn_calibration_size(n_samples: int, warn_threshold: int = 50) -> None:
    warnings.warn(
        f"Calibration set has only {n_samples} samples. "
        f"Coverage guarantees may be unreliable. [UF-W001]",
        UncertaintyFlowWarning,
        stacklevel=3,
    )
```

## Import Organization

### Standard Library Imports
```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
import warnings
```

### Third-party Imports
```python
import polars as pl
import numpy as np
```

### Local Imports
```python
from .types import PolarsInput, TargetSpec
from ..core.distribution import DistributionPrediction
```

### Order of Imports
1. Standard library (absolute imports)
2. Third-party libraries
3. Local relative imports
4. `from __future__ import annotations` at top

## Docstring Style

### Google Style Docstrings

All docstrings follow Google style format with sections:
- Summary (one line)
- Args section
- Returns section  
- Raises section
- Examples section (when helpful)

```python
def function(param: int) -> str:
    """
    Brief description of the function.

    Args:
        param: Description of the parameter

    Returns:
        Description of return value

    Raises:
        ValueError: If param is negative

    Examples:
        >>> function(5)
        'output'
    """
```

### Class Docstrings
- Include purpose and inheritance info
- Document abstract methods
- Note important class attributes

### Module Docstrings
- Brief description of module purpose
- List main exports
- Include usage examples if complex
