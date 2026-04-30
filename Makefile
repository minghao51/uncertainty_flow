.PHONY: docs-serve docs-build docs-stubs

docs-serve: docs-stubs
	uv run mkdocs serve

docs-build: docs-stubs
	uv run mkdocs build

docs-stubs:
	uv run python scripts/generate_notebook_docs.py
