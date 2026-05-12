#!/usr/bin/env python3
"""CI policy checks for security ignore metadata and warning filters."""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_FILE = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"

IGNORE_RE = re.compile(r"--ignore-vuln\s+(CVE-\d{4}-\d+)")
OWNER_RE = re.compile(r"owner=([A-Za-z0-9_.-]+)")
EXPIRES_RE = re.compile(r"expires=(\d{4}-\d{2}-\d{2})")
BROAD_WARNING_FILTER = "ignore::uncertainty_flow.utils.exceptions.UncertaintyFlowWarning"


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
                f"{CI_FILE}:{i+1} pip-audit ignores require owner=<name> and expires=YYYY-MM-DD metadata"
            )
            continue
        exp_date = dt.date.fromisoformat(expires.group(1))
        if exp_date < dt.date.today():
            errors.append(
                f"{CI_FILE}:{i+1} pip-audit ignore metadata expired on {exp_date.isoformat()}"
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


def main() -> int:
    errors = [*check_pip_audit_ignores(), *check_warning_filters()]
    if errors:
        print("Policy check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
