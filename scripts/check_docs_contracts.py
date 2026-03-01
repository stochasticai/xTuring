#!/usr/bin/env python3
"""Validate that docs reference existing CLI commands and API routes."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = PROJECT_ROOT / "src" / "xturing" / "cli"
API_FILE = CLI_DIR / "api.py"
DOC_FILES = [
    PROJECT_ROOT / "README.md",
    *(PROJECT_ROOT / "docs" / "docs").rglob("*.md"),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_cli_commands() -> Set[str]:
    commands = set()
    for path in CLI_DIR.glob("*.py"):
        content = _read(path)
        commands.update(
            re.findall(r'@click\.command\(name="([a-zA-Z0-9_-]+)"\)', content)
        )
    return commands


def _extract_api_routes() -> Set[str]:
    content = _read(API_FILE)
    routes = set(
        re.findall(
            r'@app\.(?:get|post|put|delete|patch)\("([^"]+)"\)',
            content,
        )
    )
    return routes


def _extract_doc_commands(contents: Iterable[str]) -> Set[str]:
    commands = set()
    for content in contents:
        commands.update(
            re.findall(
                r"(?m)^\s*(?:\$|>)\s*xturing\s+([a-zA-Z0-9_-]+)\b",
                content,
            )
        )
        commands.update(re.findall(r"`xturing\s+([a-zA-Z0-9_-]+)\b", content))
    return commands


def _extract_doc_api_routes(contents: Iterable[str]) -> Set[str]:
    routes = set()
    for content in contents:
        routes.update(
            re.findall(r"http://localhost:\{PORT\}(/[-a-zA-Z0-9_/.]+)", content)
        )
        routes.update(re.findall(r"`(/(?:api|health|v1[-a-zA-Z0-9_/.]*))`", content))
    return routes


def main() -> int:
    cli_commands = _extract_cli_commands()
    api_routes = _extract_api_routes()
    doc_contents = [_read(path) for path in DOC_FILES if path.exists()]

    doc_commands = _extract_doc_commands(doc_contents)
    doc_routes = _extract_doc_api_routes(doc_contents)

    unknown_commands = sorted(doc_commands - cli_commands)
    unknown_routes = sorted(doc_routes - api_routes)

    if unknown_commands or unknown_routes:
        print("Docs contract check failed.")
        if unknown_commands:
            print("Unknown CLI commands in docs:")
            for command in unknown_commands:
                print(f"  - {command}")
        if unknown_routes:
            print("Unknown API routes in docs:")
            for route in unknown_routes:
                print(f"  - {route}")
        return 1

    print("Docs contract check passed.")
    print(
        f"Validated {len(doc_commands)} CLI command reference(s) and {len(doc_routes)} API route reference(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
