# Tech Stack

## Overview
Uncertainty Flow is a Python library for probabilistic forecasting and uncertainty quantification, built with a focus on Polars-native data handling and scikit-learn compatibility.

## Core Language & Runtime

**Python**: 3.11+
- Requires Python >=3.11
- Type hints used throughout with `from __future__ import annotations`
- Modern Python patterns (ABC, abstractmethod, TYPE_CHECKING)

## Key Dependencies

### Data Processing
- **polars** (>=0.20.0) - Primary data frame library
  - Used for all I/O operations
  - LazyFrame support for lazy evaluation
  - Native integration throughout the codebase

- **numpy** (>=1.24.0) - Numerical computing
  - Array operations and numerical routines
  - Statistical computations
  - Backend for some model operations

### Machine Learning
- **scikit-learn** (>=1.3.0) - ML framework
  - Base model interfaces (RegressorMixin)
  - Preprocessing (StandardScaler)
  - Model evaluation utilities

- **scipy** (>=1.11.0) - Scientific computing
  - Optimization routines
  - Statistical distributions
  - Numerical algorithms

### Optional Dependencies

**Development** (dev extra):
- **pytest** (>=7.4.0, >=9.0.2) - Testing framework
- **pytest-cov** (>=4.1.0, >=7.0.0) - Coverage reporting
- **ruff** (>=0.1.0) - Fast Python linter
- **matplotlib** (>=3.7.0) - Plotting library

**PyTorch** (torch extra):
- **torch** (>=2.0.0) - Deep learning framework
  - Alternative backend for DeepQuantileNetTorch
  - GPU acceleration support

## Build System

- **hatchling** - Build backend
  - Modern Python packaging tool
  - Specified in pyproject.toml

## Development Tools

### Linting & Formatting
- **ruff** - Fast Python linter/formatter
  - Line length: 100 characters
  - Target version: Python 3.11
  - Enabled rules: E (errors), F (pyflakes), I (import sorting), N (naming), W (warnings)
  - No rules ignored

### Testing
- **pytest** - Testing framework
  - Test discovery in `tests/` directory
  - Test files: `test_*.py`
  - Test classes: `Test*`
  - Test functions: `test_*`

## Version Control

- **Git** - Version control system
- **uv** - Package manager (recommended for dependency management)

## Documentation

- Markdown-based documentation in `docs/` directory
- README.md as main entry point
- Inline docstrings (Google-style format preferred)

## Platform Support

- Cross-platform (macOS, Linux, Windows)
- Developed primarily on macOS (Darwin 25.3.0)

## Dependency Philosophy

- Minimal dependencies for core functionality
- Optional extras for extended features (torch, dev tools)
- Polars-first approach for data handling
- Scikit-learn compatibility for model interoperability
