"""Generate docs/notebooks/ stub pages from marimo notebook .py files.

marimo .py notebooks are the source of truth. This script reads each
notebook, extracts its title, and generates thin mkdocs pages with
a !marimo_file directive (rendered by mkdocs-marimo plugin) + molab badge
+ local run command.
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs" / "notebooks"
GITHUB_REPO = "minghao51/uncertainty_flow"


def extract_title_from_notebook(path: Path) -> tuple[str, str]:
    """Extract title and short description from a marimo notebook."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "md":
            if node.args:
                raw = ast.literal_eval(node.args[0])
                segments = raw.split("\n\n", 2)
                title = (
                    re.sub(r"^#\s+", "", segments[0]).strip()
                    if segments
                    else path.stem.replace("_", " ").title()
                )
                desc = segments[1].strip().replace("\n", " ") if len(segments) > 1 else title
                return title, desc
    return path.stem.replace("_", " ").title(), ""


def slug_from_filename(py_path: Path) -> str:
    stem = py_path.stem  # e.g. "01_quick_start_conformal_regression"
    parts = stem.split("_", 1)
    num = parts[0]  # "01"
    name = parts[1] if len(parts) > 1 else stem
    slug = re.sub(r"_+", "-", name)
    return f"{num}-{slug}"


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(NOTEBOOKS_DIR.glob("[0-9]*.py"))

    entries = []
    for nb in notebooks:
        slug = slug_from_filename(nb)
        title, desc = extract_title_from_notebook(nb)
        entries.append((slug, nb.name, title, desc))

        page = f"""# {title}

{desc}

!marimo_file ../../notebooks/{nb.name}

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/{GITHUB_REPO}/blob/main/notebooks/{nb.name})

```bash
uv run marimo run notebooks/{nb.name}
```
"""
        (DOCS_DIR / f"{slug}.md").write_text(page.lstrip())
        print(f"  generated  docs/notebooks/{slug}.md")

    rows = "\n".join(f"| [{title}]({slug}.md) | {desc} |" for slug, _, title, desc in entries)
    index = f"""# Interactive Notebooks

marimo notebooks are the **source of truth** for all examples. Each `.py` notebook
is auto-converted to interactive HTML during CI/CD.

| Notebook | Description |
|----------|-------------|
{rows}

## Run Locally

```bash
uv sync --extra dev
uv run marimo run notebooks/{notebooks[0].name}
```
"""
    (DOCS_DIR / "index.md").write_text(index.lstrip())
    print("  generated  docs/notebooks/index.md")


if __name__ == "__main__":
    main()
