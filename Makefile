.PHONY: notebooks notebooks-staged docs docs-preview

notebooks:
	uv run quarto render notebooks/

notebooks-staged:
	@changed=$$(git diff --cached --name-only -- 'notebooks/*.qmd'); \
	if [ -n "$$changed" ]; then \
		echo "Re-rendering changed notebooks: $$changed"; \
		for f in $$changed; do uv run quarto render "$$f"; done; \
		git add notebooks/_freeze/ && git add -f docs/notebooks/html/; \
	fi

docs: notebooks
	uv run python scripts/generate_notebook_docs.py
	uv run mkdocs build

docs-preview: notebooks
	uv run python scripts/generate_notebook_docs.py
	uv run mkdocs serve
