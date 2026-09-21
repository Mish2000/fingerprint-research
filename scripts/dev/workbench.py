"""Local hybrid demo: explicit preparation, readiness, owned start/stop.

Run using the prepared Conda environment's Python. No global PATH changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fpbench.runtime_config import load_environment, validate_demo_databases


def executable(kind: str) -> str:
    explicit = os.getenv(f"FPBENCH_{kind.upper()}")
    if explicit:
        if not Path(explicit).is_file():
            raise RuntimeError(f"Configured {kind} executable does not exist")
        return explicit
    prefix = Path(sys.prefix)
    candidates = {
        "python": [Path(sys.executable)],
        "java": [prefix / "Library/lib/jvm/bin/java.exe", prefix / "bin/java"],
        "maven": [prefix / "Library/bin/mvn.cmd", prefix / "bin/mvn"],
        "docker": [Path(os.getenv("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"],
    }
    for path in candidates.get(kind, []):
        if path.is_file():
            return str(path)
    found = shutil.which({"maven": "mvn"}.get(kind, kind))
    if not found:
        raise RuntimeError(f"{kind} is missing; activate the prepared environment or set FPBENCH_{kind.upper()}")
    return found


def runtime_directory() -> Path:
    return Path(os.getenv("FPBENCH_RUNTIME_DIR") or ROOT / ".local/runtime")


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    # The Docker credential helper and Maven are colocated with their launchers.
    env["PATH"] = os.pathsep.join([str(Path(executable("docker")).parent),
                                 str(Path(sys.prefix) / "Library/bin"),
                                 str(Path(executable("java")).parent), env.get("PATH", "")])
    env["JAVA_HOME"] = str(Path(executable("java")).parent.parent)
    env["FPBENCH_API_URL"] = f"http://127.0.0.1:{env.get('FPBENCH_API_PORT', '8000')}"
    env["FPBENCH_ENROLLMENT_DIR"] = str(runtime_directory() / "enrollments")
    env.setdefault("SCANNER_CAPTURE_DIR", str(runtime_directory() / "scanner/incoming"))
    env.setdefault("SCANNER_NORMALIZED_DIR", str(runtime_directory() / "scanner/normalized"))
    env.setdefault("SCANNER_TWAIN_RAW_DIR", str(runtime_directory() / "scanner/twain"))
    helper = ROOT / ".local/scanner/biometrika_twain_capture.exe"
    if helper.is_file():
        env.setdefault("SCANNER_TWAIN_HELPER_PATH", str(helper))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    return env


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=child_environment(), check=True,
                          text=True, capture_output=capture)


def compose(*arguments: str) -> None:
    # Values have already been loaded literally into the environment. Compose
    # uses those same values and never needs to parse a second configuration file.
    run([executable("docker"), "compose", "--project-name", os.environ["COMPOSE_PROJECT_NAME"],
         "-f", str(ROOT / "apps/api/docker-compose.yml"), *arguments])


def init_config(path: Path, *, test: bool) -> None:
    if path.exists():
        raise RuntimeError("Configuration already exists; edit it explicitly")
    biometric, identity = secrets.token_hex(24), secrets.token_hex(24)
    first_port = 55434 if test else 55432
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    text = text.replace("replace-with-another-private-value", identity).replace("replace-with-private-value", biometric)
    text = text.replace("55432", str(first_port)).replace("55433", str(first_port + 1))
    if test:
        text = text.replace("fingerprint-research-demo", "fingerprint-research-test")
        text = text.replace("8765", "8766").replace("8000", "8001").replace("5173", "5174")
    text += f"\nFPBENCH_RUNTIME_DIR={ROOT / '.local' / ('test' if test else 'demo')}\n"
    text += f"FPBENCH_PYTHON={sys.executable}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print("Private configuration created; passwords were not printed.")


def check() -> None:
    validate_demo_databases()
    project = os.getenv("COMPOSE_PROJECT_NAME", "")
    if not project.startswith("fingerprint-research-"):
        raise RuntimeError("Use an explicitly named fingerprint-research-* Compose project")
    for kind in ("python", "java", "maven", "docker", "node"):
        executable(kind)
    run([executable("docker"), "version", "--format", "{{.Server.Version}}"])
    from src.fpbench.model_resources import validate_weights
    for backbone in ("resnet18", "vit_base"):
        _, identity = validate_weights(backbone)
        print(f"{backbone}: prepared {identity['size']} bytes, checksum verified")
    if not (ROOT / "apps/ui/node_modules/vite/bin/vite.js").is_file():
        raise RuntimeError("UI dependencies are missing; run npm ci in apps/ui")
    if not (ROOT / "apps/sourceafis-service/target/sourceafis-service-0.1.0.jar").is_file():
        raise RuntimeError("SourceAFIS JAR is missing; run build-java")
    print("Prerequisites verified. No resources were downloaded and no DB was modified.")


def wait_ready(url: str, process: subprocess.Popen, *, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("A demo process exited during startup; inspect its local log")
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Readiness timeout at {url}; inspect the local service log")


def stop_process(record: dict) -> None:
    import psutil
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() != record["created"] or process.exe() != record["exe"]:
            raise RuntimeError("Process ownership changed; refusing to stop a reused PID")
        children = process.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs([process, *children], timeout=5)
        for remaining in alive:
            remaining.kill()
        _, alive = psutil.wait_procs(alive, timeout=5)
        if alive:
            raise RuntimeError("Owned processes did not stop")
    except psutil.NoSuchProcess:
        return


def start() -> None:
    import psutil
    check()
    directory = runtime_directory()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "processes.json"
    if state_path.exists():
        raise RuntimeError("Demo process state exists; run stop before starting again")
    source_port = urlsplit(os.environ["SOURCEAFIS_SERVICE_URL"]).port or 8765
    api_port = int(os.getenv("FPBENCH_API_PORT", "8000"))
    ui_port = int(os.getenv("FPBENCH_UI_PORT", "5173"))
    for port in (source_port, api_port, ui_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                raise RuntimeError(f"Port {port} is already owned by another process") from None
    env = child_environment()
    env["SOURCEAFIS_HOST"] = "127.0.0.1"
    env["SOURCEAFIS_PORT"] = str(source_port)
    commands = [
        ("java", [executable("java"), "-jar", str(ROOT / "apps/sourceafis-service/target/sourceafis-service-0.1.0.jar")],
         ROOT, f"http://127.0.0.1:{source_port}/health"),
        ("api", [executable("python"), "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
         ROOT, f"http://127.0.0.1:{api_port}/ready"),
        ("ui", [executable("node"), str(ROOT / "apps/ui/node_modules/vite/bin/vite.js"),
                "--host", "127.0.0.1", "--port", str(ui_port), "--strictPort"],
         ROOT / "apps/ui", f"http://127.0.0.1:{ui_port}/"),
    ]
    records: list[dict] = []
    try:
        compose("up", "-d", "--wait", "--wait-timeout", "90")
        for name, command, cwd, url in commands:
            with (directory / f"{name}.log").open("ab") as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    start_new_session=os.name != "nt")
            identity = psutil.Process(process.pid)
            records.append({"name": name, "pid": process.pid, "created": identity.create_time(), "exe": identity.exe()})
            state_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            wait_ready(url, process)
        print(f"Demo ready: http://127.0.0.1:{ui_port}/?tab=verify")
    except BaseException:
        for record in reversed(records):
            stop_process(record)
        state_path.unlink(missing_ok=True)
        compose("stop")
        raise


def stop() -> None:
    path = runtime_directory() / "processes.json"
    if path.exists():
        for record in reversed(json.loads(path.read_text(encoding="utf-8"))):
            stop_process(record)
        path.unlink()
    compose("stop")
    print("Owned demo services stopped; database volumes and enrollment files preserved.")


def java_only(check_only: bool) -> None:
    import psutil
    port = urlsplit(os.environ["SOURCEAFIS_SERVICE_URL"]).port or 8765
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    env = child_environment()
    env.update(SOURCEAFIS_HOST="127.0.0.1", SOURCEAFIS_PORT=str(port))
    directory = runtime_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "java-check.log").open("ab") as log:
        process = subprocess.Popen([executable("java"), "-jar", str(ROOT / "apps/sourceafis-service/target/sourceafis-service-0.1.0.jar")],
            env=env, cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    identity = psutil.Process(process.pid)
    record = {"pid": process.pid, "created": identity.create_time(), "exe": identity.exe()}
    try:
        wait_ready(f"http://127.0.0.1:{port}/health", process, timeout=20)
        print("SourceAFIS is ready.")
        if not check_only:
            process.wait()
    finally:
        stop_process(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init-config", "check", "prepare-models", "build-java", "check-java", "serve-java", "start", "stop", "db-up", "db-stop"))
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--test", action="store_true", help="Create an isolated test configuration")
    args = parser.parse_args()
    if args.command == "init-config":
        init_config(args.env_file, test=args.test)
        return
    load_environment(args.env_file)
    if args.command == "prepare-models":
        from src.fpbench.model_resources import prepare_weights
        for backbone in ("resnet18", "vit_base"):
            print(json.dumps(prepare_weights(backbone)))
    elif args.command == "build-java":
        run([executable("maven"), "clean", "verify"], cwd=ROOT / "apps/sourceafis-service")
    elif args.command == "db-up":
        validate_demo_databases()
        compose("up", "-d", "--wait", "--wait-timeout", "90")
    elif args.command == "db-stop":
        compose("stop")
    elif args.command in {"check-java", "serve-java"}:
        java_only(args.command == "check-java")
    else:
        globals()[args.command]()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        # Child command arguments never contain DB credentials.
        print(f"Workbench: {exc}", file=sys.stderr)
        sys.exit(1)
