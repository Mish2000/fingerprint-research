import hashlib
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from apps.api.enrollment_sources import EnrollmentSources
from src.fpbench import model_resources
from src.fpbench.runtime_config import configured_device, load_environment, validate_demo_databases
from scripts.dev.workbench import stop_process
from scripts.dev import workbench
from src.fpbench.local_data import configured_raw_path


def test_configuration_is_literal_and_process_values_win(tmp_path):
    path = tmp_path / "config with spaces.env"
    path.write_text('FPBENCH_DEVICE=cuda\nLITERAL="$(not-a-command) ${NO_EXPANSION}"\n')
    env = {"FPBENCH_DEVICE": "cpu"}
    load_environment(path, env)
    assert env == {"FPBENCH_DEVICE": "cpu", "LITERAL": "$(not-a-command) ${NO_EXPANSION}"}


def test_demo_requires_distinct_explicit_database_destinations():
    with pytest.raises(ValueError, match="both"):
        validate_demo_databases({"DATABASE_URL": "postgresql://127.0.0.1/a"})
    with pytest.raises(ValueError, match="distinct"):
        validate_demo_databases({"DATABASE_URL": "postgresql://127.0.0.1/a", "IDENTITY_DATABASE_URL": "postgresql://localhost/a"})


def test_device_defaults_to_cpu_without_cuda_auto_selection(monkeypatch):
    monkeypatch.delenv("FPBENCH_DEVICE", raising=False)
    assert configured_device() == "cpu"
    with pytest.raises(ValueError, match="fallback is disabled"):
        configured_device("auto")


def test_configured_raw_root_maps_paths_without_reselecting_or_writing(monkeypatch, tmp_path):
    monkeypatch.setenv("FPBENCH_DATASETS_ROOT", str(tmp_path))
    assert configured_raw_path("C:/old/repo/data/raw/sd300b/images/example.png") == tmp_path / "NIST/sd300b/images/example.png"
    assert configured_raw_path("data/raw/PolyU_Hong_Kong/example.bmp") == tmp_path / "PolyU Hong Kong/example.bmp"
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="escapes"):
        configured_raw_path("data/raw/../../../outside")


def test_missing_weights_never_invoke_download(monkeypatch, tmp_path):
    monkeypatch.setenv("FPBENCH_WEIGHTS_DIR", str(tmp_path))
    def forbidden(*args, **kwargs):
        pytest.fail("An offline loading path attempted a download")
    monkeypatch.setattr(model_resources, "urlopen", forbidden)
    with pytest.raises(FileNotFoundError, match="prepare-models"):
        model_resources.validate_weights("resnet18")


def test_corrupted_prepared_weights_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FPBENCH_WEIGHTS_DIR", str(tmp_path))
    path = tmp_path / model_resources.WEIGHTS["resnet18"][2]
    path.write_bytes(b"not the official model")
    path.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        model_resources.validate_weights("resnet18")


def test_retained_input_is_content_bound_and_path_safe(tmp_path):
    original = tmp_path / "source.png"
    original.write_bytes(b"synthetic test bytes")
    sources = EnrollmentSources(tmp_path / "inputs")
    digest = sources.save(original)
    assert digest == hashlib.sha256(original.read_bytes()).hexdigest()
    assert sources.resolve(digest).read_bytes() == original.read_bytes()
    sources.path(digest).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        sources.resolve(digest)
    with pytest.raises(ValueError, match="digest"):
        sources.remove("../../outside")


def test_stop_checks_ownership_and_terminates_child_tree():
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
    with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) as process:
        child_id = int(process.stdout.readline())
        identity = psutil.Process(process.pid)
        record = {"pid": process.pid, "created": identity.create_time(), "exe": identity.exe()}
        try:
            with pytest.raises(RuntimeError, match="ownership"):
                stop_process({**record, "created": record["created"] - 1})
            assert process.poll() is None
            stop_process(record)
            assert not psutil.pid_exists(child_id)
        finally:
            if process.poll() is None:
                stop_process(record)


@pytest.mark.parametrize("failure", [RuntimeError("Readiness timeout"), KeyboardInterrupt()])
def test_java_startup_timeout_or_cancellation_stops_the_owned_process(monkeypatch, tmp_path, failure):
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("SOURCEAFIS_SERVICE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(workbench, "runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(workbench, "child_environment", lambda: dict(__import__("os").environ))
    monkeypatch.setattr(workbench, "executable", lambda kind: sys.executable)
    actual_popen = subprocess.Popen
    children = []
    def start_dummy(command, **kwargs):
        child = actual_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child
    def fail_readiness(*args, **kwargs):
        raise failure
    monkeypatch.setattr(workbench.subprocess, "Popen", start_dummy)
    monkeypatch.setattr(workbench, "wait_ready", fail_readiness)
    with pytest.raises(type(failure)):
        workbench.java_only(True)
    assert len(children) == 1 and children[0].poll() is not None
