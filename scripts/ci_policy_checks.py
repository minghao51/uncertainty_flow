#!/usr/bin/env python3
"""CI policy checks for security ignore metadata and warning filters."""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_FILE = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"
AGENTS_FILE = ROOT / "AGENTS.md"
CHANGELOG_FILE = ROOT / "docs" / "project" / "changelog.md"

IGNORE_RE = re.compile(r"--ignore-vuln\s+(CVE-\d{4}-\d+)")
OWNER_RE = re.compile(r"owner=([A-Za-z0-9_.-]+)")
EXPIRES_RE = re.compile(r"expires=(\d{4}-\d{2}-\d{2})")
BROAD_WARNING_FILTER = "ignore::uncertainty_flow.utils.exceptions.UncertaintyFlowWarning"
AGENTS_REF_RE = re.compile(r"@\s*([./][^\s]+)")
PUBLIC_API_PATH_PREFIXES = (
    "uncertainty_flow/cli.py",
    "uncertainty_flow/benchmarking/",
    "uncertainty_flow/multimodal/",
    "uncertainty_flow/core/distribution.py",
)


def check_pip_audit_ignores() -> list[str]:
    errors: list[str] = []
    lines = CI_FILE.read_text().splitlines()
    for i, line in enumerate(lines):
        if "pip-audit" not in line:
            continue
        cves = IGNORE_RE.findall(line)
        if not cves:
            continue
        owner = OWNER_RE.search(line)
        expires = EXPIRES_RE.search(line)
        if not owner or not expires:
            errors.append(
                f"{CI_FILE}:{i + 1} pip-audit ignores require "
                f"owner=<name> and expires=YYYY-MM-DD metadata"
            )
            continue
        exp_date = dt.date.fromisoformat(expires.group(1))
        if exp_date < dt.date.today():
            errors.append(
                f"{CI_FILE}:{i + 1} pip-audit ignore metadata expired on {exp_date.isoformat()}"
            )
    return errors


def check_warning_filters() -> list[str]:
    errors: list[str] = []
    text = PYPROJECT.read_text()
    if BROAD_WARNING_FILTER in text:
        errors.append(
            f"{PYPROJECT}: broad warning suppression '{BROAD_WARNING_FILTER}' is not allowed"
        )
    return errors


def check_agents_refs_exist() -> list[str]:
    errors: list[str] = []
    text = AGENTS_FILE.read_text()
    for ref in AGENTS_REF_RE.findall(text):
        path = (ROOT / ref).resolve()
        if not path.exists():
            errors.append(f"{AGENTS_FILE}: missing referenced context file '{ref}'")
    return errors


def _changed_files_from_env() -> list[str]:
    raw = os.environ.get("CHANGED_FILES", "").strip()
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,\n]", raw) if x.strip()]


def check_public_api_has_changelog() -> list[str]:
    errors: list[str] = []
    changed = _changed_files_from_env()
    if not changed:
        return errors
    touches_public_api = any(
        any(path.startswith(prefix) for prefix in PUBLIC_API_PATH_PREFIXES) for path in changed
    )
    changelog_changed = any(path == "docs/project/changelog.md" for path in changed)
    if touches_public_api and not changelog_changed:
        errors.append(
            "Public API surface changed but docs/project/changelog.md was not updated. "
            "Add a changelog entry or include docs/project/changelog.md in CHANGED_FILES."
        )
    return errors


def main() -> int:
    errors = [
        *check_pip_audit_ignores(),
        *check_warning_filters(),
        *check_agents_refs_exist(),
        *check_public_api_has_changelog(),
    ]
    if errors:
        print("Policy check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
