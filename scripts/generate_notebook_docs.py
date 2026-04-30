"""Generate docs/notebooks/ stub pages from marimo notebook .py files.

Updates for each notebook:
  1. Static HTML export via `marimo export html --no-include-code` placed in
     docs/notebooks/html/{slug}.html
  2. mkdocs stub page with description, an iframe embedding the exported HTML,
     a molab badge, and a local run command.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs" / "notebooks"
EXPORT_DIR = DOCS_DIR / "html"
GITHUB_REPO = "minghao51/uncertainty_flow"


def extract_title_from_notebook(path: Path) -> tuple[str, str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "md":
            if node.args:
                raw = ast.literal_eval(node.args[0])
                segments = raw.split("\n\n", 2)
                title = (
                    re.sub(r"^\s*#\s+", "", segments[0]).strip()
                    if segments
                    else path.stem.replace("_", " ").title()
                )
                desc = segments[1].strip().replace("\n", " ") if len(segments) > 1 else title
                return title, desc
    return path.stem.replace("_", " ").title(), ""


def slug_from_filename(py_path: Path) -> str:
    stem = py_path.stem
    parts = stem.split("_", 1)
    num = parts[0]
    name = parts[1] if len(parts) > 1 else stem
    slug = re.sub(r"_+", "-", name)
    return f"{num}-{slug}"


def export_notebook_html(nb: Path, slug: str) -> tuple[bool, str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{slug}.html"
    if out_path.exists():
        print(f"  export exists  html/{slug}.html (skipping)")
        return True, ""
    print(f"  exporting     html/{slug}.html ...", end=" ")
    sys.stdout.flush()
    result = subprocess.run(
        [
            "uv",
            "run",
            "marimo",
            "export",
            "html",
            nb.name,
            "--no-include-code",
            "-o",
            str(out_path),
            "--no-sandbox",
        ],
        cwd=str(NOTEBOOKS_DIR),
        capture_output=True,
        text=True,
    )
    had_errors = "error" in result.stderr.lower() or "traceback" in result.stderr.lower()
    note = ""
    if out_path.exists() and out_path.stat().st_size > 1000:
        print("OK" if not had_errors else "OK (with cell errors)")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[:3]:
                print(f"    {line}")
        if had_errors:
            note = (
                "Some cells failed to execute during static export. "
                "Use the molab badge below for full interactivity."
            )
    else:
        print("FAILED")
        if result.stderr.strip():
            print(f"    {result.stderr.strip()[:500]}")
        return False, ""
    return True, note


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(NOTEBOOKS_DIR.glob("[0-9]*.py"))

    entries = []
    for nb in notebooks:
        slug = slug_from_filename(nb)
        title, desc = extract_title_from_notebook(nb)
        entries.append((slug, nb.name, title, desc))

        export_ok, export_note = export_notebook_html(nb, slug)
        export_path = f"../html/{slug}.html"

        note_block = f"\n> **Note:** {export_note}\n" if export_note else ""

        if export_ok:
            page = f"""# {title}

{desc}
{note_block}
<div style="margin: 0 -0.8rem">
  <iframe
    src="{export_path}"
    style="width: 100%; height: 600px; border: 1px solid
      var(--md-default-fg-color--lightest); border-radius: 4px;"
    loading="lazy"
  ></iframe>
</div>

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/{GITHUB_REPO}/blob/main/notebooks/{nb.name})

```bash
uv run marimo run notebooks/{nb.name}
```
"""
        else:
            page = f"""# {title}

{desc}

> **Note:** The notebook could not be pre-rendered as static HTML. Use the
> links below to run it interactively.

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
is executed and exported as static HTML during CI/CD.

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
