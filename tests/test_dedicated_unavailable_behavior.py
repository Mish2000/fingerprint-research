from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import apps.api.main as api_main
from apps.api.service import MatchService, MethodUnavailableError


def _write_png(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok, "cv2.imencode failed"
    path.write_bytes(buf.tobytes())
    return path


class _BrokenDedicatedMatcher:
    def __init__(self, *args, **kwargs):
        raise FileNotFoundError("Descriptor checkpoint not found: missing-test-checkpoint.pth")


def test_dedicated_unavailable_is_reported_without_killing_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = MatchService(dedicated_factory=_BrokenDedicatedMatcher)

    first = _write_png(tmp_path / "a.png", 31)
    second = _write_png(tmp_path / "b.png", 32)

    availability = service.method_availability()
    assert availability["classic_orb"]["available"] is True
    assert availability["classic_gftt_orb"]["available"] is True
    assert availability["dedicated"]["available"] is False
    assert "Descriptor checkpoint not found" in str(availability["dedicated"]["error"])

    with pytest.raises(MethodUnavailableError, match="Dedicated \\(Patch AI\\)"):
        service.match(
            method="dedicated",
            path_a=str(first),
            path_b=str(second),
            threshold=None,
            return_overlay=False,
            capture_a="plain",
            capture_b="plain",
            filename_a=first.name,
            filename_b=second.name,
        )

    monkeypatch.setattr(api_main, "_service", service)
    monkeypatch.setattr(api_main, "_service_init_error", None)
    payload = api_main.health()

    assert payload["ok"] is True
    assert payload["methods"]["dedicated"]["available"] is False
    assert "missing-test-checkpoint.pth" in payload["methods"]["dedicated"]["error"]
