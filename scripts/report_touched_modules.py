#!/usr/bin/env python3
"""Report touched uncertainty_flow modules in the current branch."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _base_ref() -> str:
    try:
        _git(["rev-parse", "--verify", "origin/main"])
        return "origin/main...HEAD"
    except subprocess.CalledProcessError:
        return "HEAD~1...HEAD"


def main() -> int:
    base = _base_ref()
    out = _git(["diff", "--name-only", base])
    files = [
        f
        for f in out.splitlines()
        if f.startswith("uncertainty_flow/") and f.endswith(".py")
    ]
    if not files:
        print("No touched uncertainty_flow Python modules detected")
        return 0
    print("Touched uncertainty_flow modules:")
    for file in sorted(files):
        print(f"- {file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
