"""Generate docs/notebooks/ stub pages from Quarto .qmd notebook files.

Reads YAML frontmatter from each .qmd file and generates MkDocs stub pages
with iframe embeds pointing to Quarto-rendered HTML.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs" / "notebooks"
GITHUB_REPO = "minghao51/uncertainty_flow"


def get_base_path() -> str:
    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    text = mkdocs_path.read_text()
    for line in text.splitlines():
        if line.strip().startswith("site_url:"):
            url = line.split(":", 1)[1].strip()
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path = parsed.path.strip("/")
            return f"/{path}" if path else ""
    return ""


def parse_frontmatter(path: Path) -> dict:
    text = path.read_text()
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    yaml_text = text[3:end].strip()
    meta = {}
    for line in yaml_text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().strip('"').strip("'")
            if val:
                meta[key.strip()] = val
    return meta


def slug_from_filename(qmd_path: Path) -> str:
    stem = qmd_path.stem
    parts = stem.split("_", 1)
    num = parts[0]
    name = parts[1] if len(parts) > 1 else stem
    slug = re.sub(r"_+", "-", name)
    return f"{num}-{slug}"


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "html").mkdir(parents=True, exist_ok=True)
    base = get_base_path()

    qmd_files = sorted(NOTEBOOKS_DIR.glob("[0-9]*.qmd"))
    if not qmd_files:
        print("No .qmd notebooks found in notebooks/")
        return

    entries = []
    for qmd in qmd_files:
        slug = slug_from_filename(qmd)
        meta = parse_frontmatter(qmd)
        title = meta.get("title", qmd.stem.replace("_", " ").title())
        desc = meta.get("description", title)

        html_stem = qmd.stem
        html_path = f"html/{html_stem}.html"
        html_exists = (DOCS_DIR / html_path).exists()

        if html_exists:
            note = ""
        else:
            note = (
                "\n> **Note:** Rendered HTML not yet available. "
                "Run `make notebooks` locally to generate it.\n"
            )

        page = f"""---
hide:
  - navigation
  - toc
---

# {title}

{desc}
{note}
<div class="iframe-container" id="iframe-wrapper-{qmd.stem.replace("_", "-")}">
  <div class="iframe-controls">
    <button onclick="toggleNotebookFullscreen(this)" class="md-button">Expand</button>
    <a href="{base}/notebooks/{html_path}" target="_blank" rel="noopener noreferrer" class="md-button">Open in New Tab</a>
  </div>
  <iframe src="{base}/notebooks/{html_path}" allowfullscreen loading="lazy"></iframe>
</div>

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/{GITHUB_REPO}/blob/main/notebooks/{qmd.name})

## Run Locally

```bash
uv sync --extra opinion
# See notebooks/{qmd.name} for the full tutorial
```
"""
        (DOCS_DIR / f"{slug}.md").write_text(page.lstrip())
        print(f"  generated  docs/notebooks/{slug}.md")
        entries.append((slug, qmd.name, title, desc))

    rows = "\n".join(f"| [{title}]({slug}.md) | {desc} |" for slug, _, title, desc in entries)
    index = f"""# Interactive Notebooks

Quarto `.qmd` notebooks are the **source of truth** for all examples. Each notebook
is rendered to static HTML via `quarto render` with `freeze: auto` caching.

| Notebook | Description |
|----------|-------------|
{rows}

## Run Locally

```bash
uv sync --extra opinion
make notebooks
```
"""
    (DOCS_DIR / "index.md").write_text(index.lstrip())
    print("  generated  docs/notebooks/index.md")


if __name__ == "__main__":
    main()
