"""Opt-in, local input retention for the demo's existing pairwise reranker.

Images remain outside PostgreSQL; the existing raw metadata SHA-256 links them.
No identity data, templates, or vectors are written into this directory.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile


class EnrollmentSources:
    def __init__(self, root: Path):
        self.root = root

    def path(self, digest: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid enrollment image digest")
        return self.root / f"{digest}.image"

    def save(self, source: Path) -> str:
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        destination = self.path(digest)
        self.root.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        self.resolve(digest)
        return digest

    def resolve(self, digest: str) -> Path:
        path = self.path(digest)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Retained enrollment image checksum mismatch")
        return path

    def remove(self, digest: str) -> None:
        self.path(digest).unlink(missing_ok=True)
