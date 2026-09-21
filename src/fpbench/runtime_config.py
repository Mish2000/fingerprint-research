"""Explicit local configuration shared by the API and development commands."""
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def load_environment(path: str | Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Load literal KEY=value entries; the existing process environment wins.

    No shell evaluation, interpolation, export statements, or implicit search for
    files. Optional matching single/double quotes enclose a literal value.
    """
    target = os.environ if environ is None else environ
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise ValueError(f"Invalid configuration entry on line {number}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed configuration quote on line {number}")
            value = value[1:-1]
        target.setdefault(key, value)


def configured_device(device: str | None = None) -> str:
    selected = device or os.getenv("FPBENCH_DEVICE", "cpu")
    if selected not in {"cpu", "cuda"}:
        raise ValueError("FPBENCH_DEVICE must be cpu or cuda; device fallback is disabled")
    return selected


def validate_demo_databases(environ: MutableMapping[str, str] | None = None) -> None:
    settings = os.environ if environ is None else environ
    urls = [settings.get(key, "") for key in ("DATABASE_URL", "IDENTITY_DATABASE_URL")]
    if not all(urls):
        raise ValueError("The demo requires both DATABASE_URL and IDENTITY_DATABASE_URL")
    parsed = [urlsplit(url) for url in urls]
    if any(p.scheme not in {"postgres", "postgresql"} or p.hostname not in {"127.0.0.1", "localhost"} for p in parsed):
        raise ValueError("The local demo requires explicit loopback PostgreSQL URLs")
    identities = [("loopback", p.port or 5432, p.path) for p in parsed]
    if identities[0] == identities[1]:
        raise ValueError("The demo requires two distinct databases")
