#!/usr/bin/env python3
"""Run configured table-reproduction commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, argv: list[str], dry_run: bool) -> None:
    command = [sys.executable, *argv]
    print(f"[RUN] {name}: {' '.join(command)}")
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce OTDQ-Eval tables from a JSON config.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reproduction.json")
    parser.add_argument("--only", nargs="*", default=None, help="Optional step names to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    steps = config.get("table_steps", [])
    selected = set(args.only or [])
    for step in steps:
        name = step["name"]
        if selected and name not in selected:
            continue
        run_step(name, step["argv"], args.dry_run)


if __name__ == "__main__":
    main()
