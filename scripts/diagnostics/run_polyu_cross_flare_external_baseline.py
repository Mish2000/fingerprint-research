"""Phase 4C.1 external FLARE zero-shot baseline for PolyU Cross.

This diagnostic integrates the public FLARE implementation as an isolated
external dependency. It does not copy FLARE internals into fpbench, does not
fine-tune, does not add P2/enhancement/fusion, and keeps TEST closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.run_polyu_cross_zero_shot import (
    git_info,
    safe_pkg_version,
    sha256_file,
    utc_now,
)
from scripts.diagnostics import run_polyu_cross_local_correspondence_feasibility as local
from scripts.diagnostics import run_polyu_cross_pair_conditioned_correspondence as pc
from scripts.diagnostics import run_polyu_cross_representation_alignment as base
from src.fpbench.datasets.polyu_cross_pairs import load_polyu_cross_pairs


RUN_SCHEMA_VERSION = "polyu_cross_flare_external_baseline_v0"
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_PHASE4B1_DIR = "artifacts/reports/diagnostics/polyu_cross_representation_alignment_v0"
DEFAULT_FLARE_REPO = "artifacts/external/FLARE"
DEFAULT_OUTDIR = "artifacts/reports/diagnostics/polyu_cross_flare_external_baseline_v0"
DEFAULT_P2_REFERENCE = "artifacts/reports/diagnostics/polyu_cross_sourceafis_readability_ladder_v0/ladder_metrics.csv"

FLARE_REPO_URL = "https://github.com/Yu-Yy/FLARE.git"
FLARE_EXPECTED_WEIGHTS = {
    "FDD": {
        "filename": "desc_model.pth.tar",
        "relative_path": "model_weights/desc_model.pth.tar",
        "source_url": "https://drive.google.com/file/d/1zvAI57L0TDC7q6kQgNh5_DwSbicjJ4hs/view?usp=drive_link",
    },
    "RegressionPose": {
        "filename": "RegressionPose.pth",
        "relative_path": "model_weights/RegressionPose.pth",
        "source_url": "https://drive.google.com/file/d/1AXpN8GBSqhlIXDilqPLfZf9n4Dc0pEpj/view?usp=drive_link",
    },
    "VotingPose": {
        "filename": "VotingPose.pth",
        "relative_path": "model_weights/VotingPose.pth",
        "source_url": "https://drive.google.com/file/d/1Zg4duNJ8mg-fkTACTzpPPK7DgNb9NRvA/view?usp=drive_link",
    },
}

E0 = "E0_existing_reference"
E1 = "E1_FLARE_RegressionPose"
E2 = "E2_FLARE_VotingPose"
CONDITION_TO_POSE = {E1: "RegressionPose", E2: "VotingPose"}
POSE_TO_SCRIPT = {"RegressionPose": "extract_RegressionPose.py", "VotingPose": "extract_VotingPose.py"}

CLCB = "contactless_to_contact_based"
CLCL_SAME = "contactless_to_contactless_same_session"
CLCL_CROSS = "contactless_to_contactless_cross_session"
CBCB_SAME = "contact_based_to_contact_based_same_session"
CBCB_CROSS = "contact_based_to_contact_based_cross_session"
CONTROL_PROTOCOLS = (CLCB, CLCL_SAME, CLCL_CROSS, CBCB_SAME, CBCB_CROSS)
WITHIN_PROTOCOLS = (CLCL_SAME, CLCL_CROSS, CBCB_SAME, CBCB_CROSS)

CONTACTLESS = base.CONTACTLESS
CONTACT = base.CONTACT
TRAIN = base.TRAIN
VAL = base.VAL

EXISTING_CUSTOM_BASELINE_AUC = 0.574156
BASELINE_RETRIEVAL = pc.BASELINE_HARD_MNN_RETRIEVAL


class FlareBaselineError(RuntimeError):
    """Raised for Phase 4C.1 protocol, adapter, or dependency failures."""


@dataclass(frozen=True)
class FlareConfig:
    seed: int = 1341
    eval_max_pos: int = 400
    eval_neg_per_pos: int = 3
    score_range_min: float = 1e-6
    unique_score_count_min: int = 10
    strong_auc_threshold: float = 0.70
    material_auc_threshold: float = 0.60
    material_retrieval_mrr_gain: float = 0.03
    within_same_session_min_auc: float = 0.70
    within_cross_session_min_auc: float = 0.60
    max_failure_fraction: float = 0.01
    selection_auc_tie_margin: float = 0.01
    poll_interval_seconds: float = 0.25


@dataclass
class CommandResult:
    returncode: int
    elapsed_seconds: float
    log_path: Path
    peak_gpu_memory_mib: Optional[float]
    baseline_gpu_memory_mib: Optional[float]
    stdout: str


@dataclass
class StageData:
    stage: str
    split: str
    images: pd.DataFrame
    identity_ids: list[str]
    pair_bundle: dict[str, pd.DataFrame]
    retrieval_table: pd.DataFrame
    resolved_root: Any
    pair_counts: dict[str, dict[str, int]]


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _run_git(flare_repo: Path, args: list[str]) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=str(flare_repo), capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def flare_provenance(flare_repo: Path) -> dict[str, Any]:
    readme = Path(flare_repo) / "README.md"
    license_file = Path(flare_repo) / "LICENSE"
    return {
        "repository_url": FLARE_REPO_URL,
        "local_path": str(Path(flare_repo).resolve()),
        "commit_sha": _run_git(flare_repo, ["rev-parse", "HEAD"]),
        "branch": _run_git(flare_repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run_git(flare_repo, ["remote", "get-url", "origin"]),
        "is_dirty": bool((_run_git(flare_repo, ["status", "--porcelain"]) or "").strip()),
        "readme_sha256": sha256_file(readme) if readme.exists() else None,
        "license_file_sha256": sha256_file(license_file) if license_file.exists() else None,
        "license_research_use_notes": {
            "readme_notice": "README states Academic Research License / academic and educational use only / commercial use prohibited.",
            "license_file": "Repository also contains a LICENSE file with Apache-2.0 text; record both upstream notices without modifying them.",
        },
        "expected_weights": FLARE_EXPECTED_WEIGHTS,
        "upstream_entrypoints": {
            "RegressionPose": "extract_RegressionPose.py -f <dataset> -g <gpu>",
            "VotingPose": "extract_VotingPose.py -f <dataset> -g <gpu>",
            "FDD": "extract_FDD.py -f <dataset> -g <gpu> -p <RegressionPose|VotingPose>",
        },
    }


def model_weight_inventory(flare_repo: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in FLARE_EXPECTED_WEIGHTS.items():
        path = Path(flare_repo) / spec["relative_path"]
        rows.append(
            {
                "weight_name": name,
                "expected_filename": spec["filename"],
                "path": str(path),
                "exists": bool(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else None,
                "source_url": spec["source_url"],
            }
        )
    return pd.DataFrame(rows)


def dependency_versions() -> dict[str, Any]:
    versions = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    for package in (
        "torch",
        "numpy",
        "pandas",
        "opencv-python",
        "scipy",
        "Pillow",
        "PyYAML",
        "easydict",
        "tqdm",
        "gdown",
        "scikit-learn",
    ):
        versions[package] = safe_pkg_version(package)
    try:
        import torch

        versions["torch_cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_cuda_device_count"] = int(torch.cuda.device_count())
        versions["torch_cuda_device_0"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions["torch_cuda_version"] = torch.version.cuda
    except Exception as exc:
        versions["torch_error"] = repr(exc)
    return versions


def validate_flare_dependency(flare_repo: Path) -> list[str]:
    failures: list[str] = []
    required = ["README.md", "extract_FDD.py", "extract_RegressionPose.py", "extract_VotingPose.py"]
    for rel in required:
        if not (Path(flare_repo) / rel).exists():
            failures.append(f"missing_upstream_file:{rel}")
    weights = model_weight_inventory(flare_repo)
    missing = weights[~weights["exists"].astype(bool)]
    for _, row in missing.iterrows():
        failures.append(f"missing_weight:{row['weight_name']}:{row['path']}")
    return failures


def canonical_hashes(manifest_dir: Path) -> dict[str, Any]:
    files = {
        "manifest_csv": Path(manifest_dir) / "manifest.csv",
        "pairs_train_csv": Path(manifest_dir) / "pairs_train.csv",
        "pairs_val_csv": Path(manifest_dir) / "pairs_val.csv",
        "pairs_test_csv_integrity_only": Path(manifest_dir) / "pairs_test.csv",
    }
    return {name: sha256_file(path) for name, path in files.items() if path.exists()}


def load_stage_data(
    *,
    stage: str,
    manifest_dir: Path,
    phase4b1_dir: Path,
    polyu_root: Optional[str],
    cfg: FlareConfig,
) -> StageData:
    if stage == "inner_dev":
        images_all, resolved_root = local.load_manifest_for_splits(
            manifest_dir, polyu_root=polyu_root, splits=[TRAIN]
        )
        train_ids = sorted(images_all["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
        inner_split = local.load_fixed_inner_split(train_ids, phase4b1_dir, split_seed=cfg.seed)
        ids = inner_split["inner_dev"]
        images = images_all[images_all["finger_unit_id"].astype(str).isin(set(ids))].copy()
        clcb = canonical_clcb_pairs(manifest_dir / "pairs_train.csv", ids, stage=stage)
        split = TRAIN
    elif stage == "official_val":
        images, resolved_root = local.load_manifest_for_splits(manifest_dir, polyu_root=polyu_root, splits=[VAL])
        ids = sorted(images["finger_unit_id"].astype(str).unique().tolist(), key=base.natural_identity_key)
        clcb = load_polyu_cross_pairs(manifest_dir / "pairs_val.csv").copy()
        clcb["protocol_id"] = CLCB
        split = VAL
    else:
        raise FlareBaselineError(f"Unknown stage {stage!r}")

    if images.empty:
        raise FlareBaselineError(f"No images for stage {stage}")
    pair_bundle = {
        CLCB: clcb.reset_index(drop=True),
        CLCL_SAME: base.build_within_modality_pairs(
            images,
            ids,
            protocol=CLCL_SAME,
            modality=CONTACTLESS,
            relation="same",
            split_name=stage,
            max_pos=cfg.eval_max_pos,
            neg_per_pos=cfg.eval_neg_per_pos,
            seed=cfg.seed,
        ),
        CLCL_CROSS: base.build_within_modality_pairs(
            images,
            ids,
            protocol=CLCL_CROSS,
            modality=CONTACTLESS,
            relation="cross",
            split_name=stage,
            max_pos=cfg.eval_max_pos,
            neg_per_pos=cfg.eval_neg_per_pos,
            seed=cfg.seed,
        ),
        CBCB_SAME: base.build_within_modality_pairs(
            images,
            ids,
            protocol=CBCB_SAME,
            modality=CONTACT,
            relation="same",
            split_name=stage,
            max_pos=cfg.eval_max_pos,
            neg_per_pos=cfg.eval_neg_per_pos,
            seed=cfg.seed,
        ),
        CBCB_CROSS: base.build_within_modality_pairs(
            images,
            ids,
            protocol=CBCB_CROSS,
            modality=CONTACT,
            relation="cross",
            split_name=stage,
            max_pos=cfg.eval_max_pos,
            neg_per_pos=cfg.eval_neg_per_pos,
            seed=cfg.seed,
        ),
    }
    pair_counts = validate_pair_bundle(pair_bundle)
    return StageData(
        stage=stage,
        split=split,
        images=images.reset_index(drop=True),
        identity_ids=ids,
        pair_bundle=pair_bundle,
        retrieval_table=base.build_retrieval_table(images, ids),
        resolved_root=resolved_root,
        pair_counts=pair_counts,
    )


def canonical_clcb_pairs(pairs_csv: Path, identity_ids: Iterable[str], *, stage: str) -> pd.DataFrame:
    ids = {str(x) for x in identity_ids}
    pairs = load_polyu_cross_pairs(pairs_csv).copy()
    required = {"finger_unit_a", "finger_unit_b", "sample_uid_a", "sample_uid_b"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise FlareBaselineError(f"Canonical CL->CB pairs missing columns {missing}")
    pairs = pairs[
        pairs["finger_unit_a"].astype(str).isin(ids) & pairs["finger_unit_b"].astype(str).isin(ids)
    ].copy()
    if pairs.empty:
        raise FlareBaselineError(f"No canonical CL->CB pairs remain after filtering {pairs_csv} to {stage} identities")
    pairs["protocol_id"] = CLCB
    return pairs.sort_values("_row_order", kind="mergesort").reset_index(drop=True)


def validate_pair_bundle(pair_bundle: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for protocol in CONTROL_PROTOCOLS:
        df = pair_bundle.get(protocol)
        if df is None or df.empty:
            raise FlareBaselineError(f"Missing or empty pair frame for {protocol}")
        labels = df["label"].astype(int)
        out[protocol] = {
            "pair_count": int(len(df)),
            "genuine_count": int((labels == 1).sum()),
            "impostor_count": int((labels == 0).sum()),
            "unique_a": int(df["sample_uid_a"].astype(str).nunique()),
            "unique_b": int(df["sample_uid_b"].astype(str).nunique()),
        }
        if out[protocol]["genuine_count"] == 0 or out[protocol]["impostor_count"] == 0:
            raise FlareBaselineError(f"Protocol {protocol} lacks both classes: {out[protocol]}")
    return out


def staged_filename(sample_uid: str, raw_path: str) -> str:
    suffix = Path(str(raw_path)).suffix.lower()
    if suffix not in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        suffix = ".png"
    return f"{sample_uid}{suffix}"


def link_or_copy(src: Path, dst: Path, *, allow_copy: bool) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError as symlink_error:
        if not allow_copy:
            raise FlareBaselineError(f"Could not hardlink or symlink {src} -> {dst}: {symlink_error}") from symlink_error
    shutil.copy2(src, dst)
    return "temporary_copy"


def build_adapter_stage(
    *,
    stage_data: StageData,
    stage_dir: Path,
    allow_copy: bool,
) -> pd.DataFrame:
    image_root = Path(stage_dir) / "image"
    rows: list[dict[str, Any]] = []
    for role in ("query", "gallery"):
        role_dir = image_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for row in stage_data.images.itertuples(index=False):
            src = Path(str(getattr(row, "resolved_path")))
            if not src.exists():
                raise FlareBaselineError(f"Missing source image for adapter: {src}")
            filename = staged_filename(str(getattr(row, "sample_uid")), str(getattr(row, "resolved_path")))
            dst = role_dir / filename
            link_type = link_or_copy(src, dst, allow_copy=allow_copy)
            rows.append(
                {
                    "stage": stage_data.stage,
                    "role": role,
                    "sample_uid": str(getattr(row, "sample_uid")),
                    "finger_unit_id": str(getattr(row, "finger_unit_id")),
                    "modality": str(getattr(row, "modality")),
                    "session_id": str(getattr(row, "session_id")),
                    "raw_path": str(src),
                    "staged_filename": filename,
                    "staged_relative_path": str(Path("image") / role / filename),
                    "link_type": link_type,
                    "copied_biometric_image": link_type == "temporary_copy",
                }
            )
    return pd.DataFrame(rows)


def query_gpu_memory_mib() -> Optional[float]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        first = proc.stdout.strip().splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def run_external_command(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    poll_interval_seconds: float,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = query_gpu_memory_mib()
    peak = baseline
    stop_event = threading.Event()

    def _poll() -> None:
        nonlocal peak
        while not stop_event.is_set():
            value = query_gpu_memory_mib()
            if value is not None:
                peak = value if peak is None else max(float(peak), float(value))
            time.sleep(float(poll_interval_seconds))

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    stop_event.set()
    poller.join(timeout=2)
    stdout = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(stdout, encoding="utf-8", errors="replace")
    return CommandResult(
        returncode=int(proc.returncode),
        elapsed_seconds=float(elapsed),
        log_path=log_path,
        peak_gpu_memory_mib=peak,
        baseline_gpu_memory_mib=baseline,
        stdout=stdout,
    )


def parse_average_time(log_text: str) -> Optional[float]:
    matches = re.findall(r"Average time for each image is\s+([0-9.eE+-]+)s(?:/sample)?", log_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_matching_time(log_text: str) -> tuple[Optional[float], Optional[float]]:
    match = re.search(r"matching consumes:\s*([0-9.eE+-]+)s,\s*speed:\s*([0-9.eE+-]+)/pair", log_text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def expected_stem(row: pd.Series) -> str:
    return str(row["sample_uid"])


def inspect_flare_outputs(stage_dir: Path, pose: str, images: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = {
        "expected_role_images": int(len(images) * 2),
        "pose_missing": 0,
        "pose_nonfinite": 0,
        "descriptor_missing": 0,
        "descriptor_nonfinite": 0,
    }
    for role in ("query", "gallery"):
        for _, image_row in images.iterrows():
            uid = expected_stem(image_row)
            pose_path = Path(stage_dir) / pose / role / f"{uid}.txt"
            desc_path = Path(stage_dir) / f"FDD_feat_{pose}" / role / f"{uid}.pkl"
            pose_exists = pose_path.exists()
            desc_exists = desc_path.exists()
            pose_finite = False
            desc_finite = False
            feature_shape = ""
            mask_shape = ""
            if pose_exists:
                try:
                    pose_arr = np.loadtxt(pose_path)
                    pose_finite = bool(np.asarray(pose_arr).size >= 3 and np.isfinite(pose_arr).all())
                except Exception:
                    pose_finite = False
            if desc_exists:
                try:
                    with desc_path.open("rb") as handle:
                        payload = pickle.load(handle)
                    feat = np.asarray(payload.get("feature"))
                    mask = np.asarray(payload.get("mask"))
                    feature_shape = "x".join(map(str, feat.shape))
                    mask_shape = "x".join(map(str, mask.shape))
                    desc_finite = bool(feat.size and mask.size and np.isfinite(feat).all() and np.isfinite(mask).all())
                except Exception:
                    desc_finite = False
            summary["pose_missing"] += int(not pose_exists)
            summary["pose_nonfinite"] += int(pose_exists and not pose_finite)
            summary["descriptor_missing"] += int(not desc_exists)
            summary["descriptor_nonfinite"] += int(desc_exists and not desc_finite)
            rows.append(
                {
                    "role": role,
                    "sample_uid": uid,
                    "modality": str(image_row["modality"]),
                    "pose_path": str(pose_path),
                    "pose_exists": pose_exists,
                    "pose_finite": pose_finite,
                    "descriptor_path": str(desc_path),
                    "descriptor_exists": desc_exists,
                    "descriptor_finite": desc_finite,
                    "feature_shape": feature_shape,
                    "mask_shape": mask_shape,
                }
            )
    return pd.DataFrame(rows), summary


def load_score_matrix(stage_dir: Path, pose: str) -> pd.DataFrame:
    score_csv = Path(stage_dir) / f"FDD_feat_{pose}" / "score_FDD.csv"
    if not score_csv.exists():
        raise FlareBaselineError(f"Missing FLARE score matrix: {score_csv}")
    return pd.read_csv(score_csv, index_col=0)


def score_pair_frame_from_matrix(df: pd.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
    scores = np.full(len(df), np.nan, dtype=np.float64)
    columns = {str(c): c for c in matrix.columns}
    index = {str(i): i for i in matrix.index}
    for pos, row in enumerate(df.itertuples(index=False)):
        a = f"{getattr(row, 'sample_uid_a')}.pkl"
        b = f"{getattr(row, 'sample_uid_b')}.pkl"
        if a not in index or b not in columns:
            continue
        value = matrix.loc[index[a], columns[b]]
        scores[pos] = float(value)
    return scores


def _score_summary(scores: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "score_min": float("nan"),
            "score_max": float("nan"),
            "score_range": float("nan"),
            "unique_score_count": 0,
            "unique_score_count_rounded_6": 0,
        }
    return {
        "score_min": float(np.min(finite)),
        "score_max": float(np.max(finite)),
        "score_range": float(np.max(finite) - np.min(finite)),
        "unique_score_count": int(np.unique(finite).size),
        "unique_score_count_rounded_6": int(np.unique(np.round(finite, 6)).size),
    }


def metric_row_from_scores(
    *,
    condition: str,
    pose_method: str,
    stage: str,
    protocol: str,
    df: pd.DataFrame,
    scores: np.ndarray,
    elapsed_seconds: float,
    this_phase_evaluated: bool = True,
    reference_source: str = "",
) -> dict[str, Any]:
    labels = df["label"].astype(int).to_numpy()
    auc, eer = base.auc_eer(labels, scores)
    gen_stats = base.group_stats(scores[labels == 1])
    imp_stats = base.group_stats(scores[labels == 0])
    pair_count = int(len(df))
    row = {
        "condition": condition,
        "pose_method": pose_method,
        "stage": stage,
        "protocol": protocol,
        "pair_count": pair_count,
        "genuine_count": int((labels == 1).sum()),
        "impostor_count": int((labels == 0).sum()),
        "scored_count": int(np.isfinite(scores).sum()),
        "failed_count": int((~np.isfinite(scores)).sum()),
        "roc_auc": auc,
        "eer": eer,
        "genuine_score_mean": gen_stats["mean"],
        "genuine_score_std": gen_stats["std"],
        "genuine_score_median": gen_stats["median"],
        "impostor_score_mean": imp_stats["mean"],
        "impostor_score_std": imp_stats["std"],
        "impostor_score_median": imp_stats["median"],
        "elapsed_seconds": float(elapsed_seconds),
        "runtime_ms_per_pair": float(1000.0 * elapsed_seconds / pair_count) if pair_count else float("nan"),
        "this_phase_evaluated": bool(this_phase_evaluated),
        "reference_source": reference_source,
    }
    row.update(_score_summary(scores))
    return row


def retrieval_metrics_from_matrix(
    *,
    condition: str,
    pose_method: str,
    stage: str,
    table: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    ids = table["finger_unit_id"].astype(str).tolist()
    cl = table["cl_uid"].astype(str).tolist()
    cb = table["cb_uid"].astype(str).tolist()

    def _matrix(probe: list[str], gallery: list[str]) -> np.ndarray:
        arr = np.full((len(probe), len(gallery)), np.nan, dtype=np.float64)
        columns = {str(c): c for c in matrix.columns}
        index = {str(i): i for i in matrix.index}
        for i, puid in enumerate(probe):
            pkey = f"{puid}.pkl"
            for j, guid in enumerate(gallery):
                gkey = f"{guid}.pkl"
                if pkey in index and gkey in columns:
                    arr[i, j] = float(matrix.loc[index[pkey], columns[gkey]])
        return arr

    for probe, gallery, direction in (
        (cl, cb, "CL_probe_to_CB_gallery"),
        (cb, cl, "CB_probe_to_CL_gallery"),
    ):
        sim = _matrix(probe, gallery)
        ranks: list[int] = []
        failed_queries = 0
        for i in range(sim.shape[0]):
            row = sim[i]
            if not np.isfinite(row).all():
                failed_queries += 1
                ranks.append(sim.shape[1] + 1)
                continue
            order = np.argsort(-row, kind="mergesort")
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)
        ranks_np = np.asarray(ranks, dtype=int)
        rows.append(
            {
                "condition": condition,
                "pose_method": pose_method,
                "stage": stage,
                "direction": direction,
                "identity_count": int(len(ids)),
                "failed_query_count": int(failed_queries),
                "recall_at_1": float(np.mean(ranks_np <= 1)),
                "recall_at_5": float(np.mean(ranks_np <= min(5, len(ids)))),
                "mrr": float(np.mean(1.0 / ranks_np)),
            }
        )
    return pd.DataFrame(rows)


def e0_reference_rows(p2_reference_csv: Path) -> list[dict[str, Any]]:
    if not Path(p2_reference_csv).exists():
        return []
    df = pd.read_csv(p2_reference_csv)
    rows: list[dict[str, Any]] = []
    sub = df[(df["variant"] == "P2_robust_intensity_norm") & (df["protocol"] == CLCB)].copy()
    for _, row in sub.iterrows():
        stage = f"existing_reference_ladder_{row['split']}"
        scores = np.array([], dtype=float)
        rows.append(
            {
                "condition": E0,
                "pose_method": "SourceAFIS_P2",
                "stage": stage,
                "protocol": CLCB,
                "pair_count": int(row["pair_count"]),
                "genuine_count": int(row["genuine_count"]),
                "impostor_count": int(row["impostor_count"]),
                "scored_count": int(row["scored_count"]),
                "failed_count": int(row["failed_count"]),
                "roc_auc": float(row["roc_auc"]),
                "eer": float("nan"),
                "genuine_score_mean": float(row["genuine_score_mean"]),
                "genuine_score_std": float(row["genuine_score_std"]),
                "genuine_score_median": float(row["genuine_score_median"]),
                "impostor_score_mean": float(row["impostor_score_mean"]),
                "impostor_score_std": float(row["impostor_score_std"]),
                "impostor_score_median": float(row["impostor_score_median"]),
                "score_min": float("nan"),
                "score_max": float("nan"),
                "score_range": float("nan"),
                "unique_score_count": 0,
                "unique_score_count_rounded_6": 0,
                "elapsed_seconds": float("nan"),
                "runtime_ms_per_pair": float("nan"),
                "this_phase_evaluated": False,
                "reference_source": str(p2_reference_csv),
            }
        )
    return rows


def evaluate_condition(
    *,
    condition: str,
    stage_data: StageData,
    flare_repo: Path,
    outdir: Path,
    gpu: str,
    cfg: FlareConfig,
    keep_staging: bool,
    allow_copy_staging: bool,
) -> dict[str, Any]:
    pose = CONDITION_TO_POSE[condition]
    stage_dir = Path(tempfile.mkdtemp(prefix=f"flare_{condition}_{stage_data.stage}_", dir=str(outdir)))
    adapter = pd.DataFrame()
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    retrieval = pd.DataFrame()
    output_inventory = pd.DataFrame()
    success = False
    score_csv_path = ""
    try:
        adapter = build_adapter_stage(stage_data=stage_data, stage_dir=stage_dir, allow_copy=allow_copy_staging)
        adapter.insert(0, "pose_method", pose)
        adapter.insert(0, "condition", condition)
        pose_cmd = [sys.executable, POSE_TO_SCRIPT[pose], "-f", str(stage_dir), "-g", str(gpu)]
        pose_result = run_external_command(
            cmd=pose_cmd,
            cwd=flare_repo,
            log_path=outdir / "logs" / f"{condition}_{stage_data.stage}_pose.log",
            poll_interval_seconds=cfg.poll_interval_seconds,
        )
        pose_avg = parse_average_time(pose_result.stdout)
        runtime_rows.append(
            {
                "condition": condition,
                "pose_method": pose,
                "stage": stage_data.stage,
                "operation": "pose_extraction",
                "image_count": int(len(stage_data.images) * 2),
                "pair_count": "",
                "elapsed_seconds": pose_result.elapsed_seconds,
                "reported_seconds_per_image": pose_avg,
                "reported_seconds_per_pair": "",
                "wall_seconds_per_image": pose_result.elapsed_seconds / max(1, len(stage_data.images) * 2),
                "wall_seconds_per_pair": "",
                "peak_gpu_memory_mib": pose_result.peak_gpu_memory_mib,
                "baseline_gpu_memory_mib": pose_result.baseline_gpu_memory_mib,
                "returncode": pose_result.returncode,
                "log_path": str(pose_result.log_path),
            }
        )
        if pose_result.returncode != 0:
            failure_rows.append(command_failure_row(condition, pose, stage_data.stage, "pose_extraction", pose_result))
            return condition_result(
                condition,
                pose,
                stage_data.stage,
                adapter,
                verification_rows,
                retrieval,
                runtime_rows,
                failure_rows,
                output_inventory,
                success=False,
                score_csv_path=score_csv_path,
                staging_dir=stage_dir,
                keep_staging=keep_staging,
            )

        fdd_cmd = [sys.executable, "extract_FDD.py", "-f", str(stage_dir), "-g", str(gpu), "-p", pose]
        fdd_result = run_external_command(
            cmd=fdd_cmd,
            cwd=flare_repo,
            log_path=outdir / "logs" / f"{condition}_{stage_data.stage}_fdd.log",
            poll_interval_seconds=cfg.poll_interval_seconds,
        )
        fdd_avg = parse_average_time(fdd_result.stdout)
        match_total, match_per_pair = parse_matching_time(fdd_result.stdout)
        matrix_pairs = int((len(stage_data.images) * 2) ** 2)
        runtime_rows.append(
            {
                "condition": condition,
                "pose_method": pose,
                "stage": stage_data.stage,
                "operation": "descriptor_extraction_and_matrix_matching",
                "image_count": int(len(stage_data.images) * 2),
                "pair_count": matrix_pairs,
                "elapsed_seconds": fdd_result.elapsed_seconds,
                "reported_seconds_per_image": fdd_avg,
                "reported_seconds_per_pair": match_per_pair,
                "wall_seconds_per_image": fdd_result.elapsed_seconds / max(1, len(stage_data.images) * 2),
                "wall_seconds_per_pair": fdd_result.elapsed_seconds / max(1, matrix_pairs),
                "matching_elapsed_seconds_reported": match_total,
                "peak_gpu_memory_mib": fdd_result.peak_gpu_memory_mib,
                "baseline_gpu_memory_mib": fdd_result.baseline_gpu_memory_mib,
                "returncode": fdd_result.returncode,
                "log_path": str(fdd_result.log_path),
            }
        )
        if fdd_result.returncode != 0:
            failure_rows.append(command_failure_row(condition, pose, stage_data.stage, "descriptor_extraction", fdd_result))
            return condition_result(
                condition,
                pose,
                stage_data.stage,
                adapter,
                verification_rows,
                retrieval,
                runtime_rows,
                failure_rows,
                output_inventory,
                success=False,
                score_csv_path=score_csv_path,
                staging_dir=stage_dir,
                keep_staging=keep_staging,
            )

        output_inventory, output_summary = inspect_flare_outputs(stage_dir, pose, stage_data.images)
        output_inventory.insert(0, "stage", stage_data.stage)
        output_inventory.insert(0, "pose_method", pose)
        output_inventory.insert(0, "condition", condition)
        failure_rows.extend(output_failure_rows(condition, pose, stage_data.stage, output_summary))
        matrix = load_score_matrix(stage_dir, pose)
        score_csv_path = str(Path(stage_dir) / f"FDD_feat_{pose}" / "score_FDD.csv")
        for protocol in CONTROL_PROTOCOLS:
            df = stage_data.pair_bundle[protocol].reset_index(drop=True).copy()
            scores = score_pair_frame_from_matrix(df, matrix)
            row = metric_row_from_scores(
                condition=condition,
                pose_method=pose,
                stage=stage_data.stage,
                protocol=protocol,
                df=df,
                scores=scores,
                elapsed_seconds=float(match_total or 0.0),
            )
            verification_rows.append(row)
            failure_rows.append(score_failure_row(condition, pose, stage_data.stage, protocol, row, cfg))
        retrieval = retrieval_metrics_from_matrix(
            condition=condition,
            pose_method=pose,
            stage=stage_data.stage,
            table=stage_data.retrieval_table,
            matrix=matrix,
        )
        success = True
        return condition_result(
            condition,
            pose,
            stage_data.stage,
            adapter,
            verification_rows,
            retrieval,
            runtime_rows,
            failure_rows,
            output_inventory,
            success=success,
            score_csv_path=score_csv_path,
            staging_dir=stage_dir,
            keep_staging=keep_staging,
        )
    finally:
        if not keep_staging:
            shutil.rmtree(stage_dir, ignore_errors=True)


def command_failure_row(
    condition: str,
    pose: str,
    stage: str,
    operation: str,
    result: CommandResult,
) -> dict[str, Any]:
    tail = "\n".join(result.stdout.strip().splitlines()[-20:])
    return {
        "condition": condition,
        "pose_method": pose,
        "stage": stage,
        "failure_type": "upstream_command",
        "operation": operation,
        "protocol": "",
        "expected_count": "",
        "failed_count": 1,
        "failure_fraction": 1.0,
        "major_issue": True,
        "details": f"returncode={result.returncode}; log={result.log_path}; tail={tail[:2000]}",
    }


def output_failure_rows(condition: str, pose: str, stage: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = int(summary["expected_role_images"])
    for failure_type, key in (
        ("pose_estimation_missing", "pose_missing"),
        ("pose_estimation_nonfinite", "pose_nonfinite"),
        ("descriptor_extraction_missing", "descriptor_missing"),
        ("descriptor_extraction_nonfinite", "descriptor_nonfinite"),
    ):
        failed = int(summary.get(key, 0))
        rows.append(
            {
                "condition": condition,
                "pose_method": pose,
                "stage": stage,
                "failure_type": failure_type,
                "operation": "output_inventory",
                "protocol": "",
                "expected_count": expected,
                "failed_count": failed,
                "failure_fraction": float(failed / expected) if expected else float("nan"),
                "major_issue": bool(failed > 0),
                "details": json.dumps(summary, sort_keys=True),
            }
        )
    return rows


def score_failure_row(
    condition: str,
    pose: str,
    stage: str,
    protocol: str,
    metric: dict[str, Any],
    cfg: FlareConfig,
) -> dict[str, Any]:
    pair_count = int(metric["pair_count"])
    failed = int(metric["failed_count"])
    unique_min = min(int(cfg.unique_score_count_min), max(1, pair_count // 2))
    degenerate = not (
        math.isfinite(float(metric["score_range"]))
        and float(metric["score_range"]) >= cfg.score_range_min
        and int(metric["unique_score_count_rounded_6"]) >= unique_min
    )
    coverage_issue = failed / max(1, pair_count) > cfg.max_failure_fraction
    details = {
        "score_range": metric["score_range"],
        "unique_score_count_rounded_6": metric["unique_score_count_rounded_6"],
        "unique_min": unique_min,
        "coverage_issue": coverage_issue,
        "degenerate_distribution": degenerate,
    }
    return {
        "condition": condition,
        "pose_method": pose,
        "stage": stage,
        "failure_type": "score_join_or_distribution",
        "operation": "pair_score_join",
        "protocol": protocol,
        "expected_count": pair_count,
        "failed_count": failed,
        "failure_fraction": float(failed / max(1, pair_count)),
        "major_issue": bool(coverage_issue or degenerate),
        "details": json.dumps(details, sort_keys=True),
    }


def condition_result(
    condition: str,
    pose: str,
    stage: str,
    adapter: pd.DataFrame,
    verification_rows: list[dict[str, Any]],
    retrieval: pd.DataFrame,
    runtime_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    output_inventory: pd.DataFrame,
    *,
    success: bool,
    score_csv_path: str,
    staging_dir: Path,
    keep_staging: bool,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "pose_method": pose,
        "stage": stage,
        "success": bool(success),
        "adapter": adapter,
        "verification_rows": verification_rows,
        "retrieval": retrieval,
        "runtime_rows": runtime_rows,
        "failure_rows": failure_rows,
        "output_inventory": output_inventory,
        "score_csv_path": score_csv_path,
        "staging_dir": str(staging_dir) if keep_staging else "",
        "staging_removed": not keep_staging,
    }


def retrieval_value(retrieval: pd.DataFrame, condition: str, direction: str, stage: str = "inner_dev") -> float:
    rows = retrieval[
        (retrieval["condition"] == condition)
        & (retrieval["direction"] == direction)
        & (retrieval["stage"] == stage)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["mrr"])


def metric_value(metrics: pd.DataFrame, condition: str, protocol: str, stage: str = "inner_dev", col: str = "roc_auc") -> float:
    rows = metrics[
        (metrics["condition"] == condition)
        & (metrics["protocol"] == protocol)
        & (metrics["stage"] == stage)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][col])


def condition_gate_rows(metrics: pd.DataFrame, retrieval: pd.DataFrame, failures: pd.DataFrame, cfg: FlareConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in (E1, E2):
        clcb_auc = metric_value(metrics, condition, CLCB)
        cl_mrr = retrieval_value(retrieval, condition, "CL_probe_to_CB_gallery")
        cb_mrr = retrieval_value(retrieval, condition, "CB_probe_to_CL_gallery")
        cl_base = float(BASELINE_RETRIEVAL["CL_probe_to_CB_gallery"]["mrr"])
        cb_base = float(BASELINE_RETRIEVAL["CB_probe_to_CL_gallery"]["mrr"])
        within = {protocol: metric_value(metrics, condition, protocol) for protocol in WITHIN_PROTOCOLS}
        coverage_rows = failures[
            (failures["condition"] == condition)
            & (failures["stage"] == "inner_dev")
            & (failures["major_issue"].astype(bool))
        ]
        criteria = {
            "clcb_auc_at_least_0_70": bool(math.isfinite(clcb_auc) and clcb_auc >= cfg.strong_auc_threshold),
            "cl_to_cb_retrieval_materially_improves": bool(
                math.isfinite(cl_mrr) and cl_mrr >= cl_base + cfg.material_retrieval_mrr_gain
            ),
            "cb_to_cl_retrieval_materially_improves": bool(
                math.isfinite(cb_mrr) and cb_mrr >= cb_base + cfg.material_retrieval_mrr_gain
            ),
            "within_modality_controls_credible": bool(
                math.isfinite(within[CLCL_SAME])
                and math.isfinite(within[CBCB_SAME])
                and math.isfinite(within[CLCL_CROSS])
                and math.isfinite(within[CBCB_CROSS])
                and within[CLCL_SAME] >= cfg.within_same_session_min_auc
                and within[CBCB_SAME] >= cfg.within_same_session_min_auc
                and within[CLCL_CROSS] >= cfg.within_cross_session_min_auc
                and within[CBCB_CROSS] >= cfg.within_cross_session_min_auc
            ),
            "no_major_sample_coverage_or_extraction_issue": bool(coverage_rows.empty),
        }
        rows.append(
            {
                "condition": condition,
                "passed": all(criteria.values()),
                "criteria": criteria,
                "clcb_auc": clcb_auc,
                "retrieval_mrr": {
                    "CL_probe_to_CB_gallery": cl_mrr,
                    "CB_probe_to_CL_gallery": cb_mrr,
                    "baseline_CL_probe_to_CB_gallery": cl_base,
                    "baseline_CB_probe_to_CL_gallery": cb_base,
                },
                "within_auc": within,
                "major_issue_count": int(len(coverage_rows)),
            }
        )
    return rows


def select_val_condition(gate_rows: list[dict[str, Any]], metrics: pd.DataFrame, cfg: FlareConfig) -> Optional[str]:
    passing = [row["condition"] for row in gate_rows if row["passed"]]
    if not passing:
        return None
    if len(passing) == 1:
        return passing[0]
    aucs = {condition: metric_value(metrics, condition, CLCB) for condition in passing}
    diff = abs(float(aucs.get(E1, float("nan"))) - float(aucs.get(E2, float("nan"))))
    if E1 in passing and E2 in passing and math.isfinite(diff) and diff <= cfg.selection_auc_tie_margin:
        return E1
    return max(passing, key=lambda c: (float(aucs.get(c, float("-inf"))), -1 if c == E1 else 0))


def classify_decision(
    *,
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    failures: pd.DataFrame,
    val_metrics: pd.DataFrame,
    cfg: FlareConfig,
    dependency_failures: list[str],
    val_condition: Optional[str],
) -> dict[str, Any]:
    if dependency_failures:
        return {
            "classification": "D. INTEGRATION_INCONCLUSIVE",
            "primary_reason": "FLARE dependency, model-weight, or upstream-file requirements were not satisfied.",
            "dependency_failures": dependency_failures,
            "inner_dev_gate": [],
            "selected_official_val_condition": None,
            "official_val_gate": {"opened": False, "reason": "Dependency gate failed."},
            "test_gate": {"opened": False, "reason": "TEST remains prohibited for Phase 4C.1."},
        }
    gate_rows = condition_gate_rows(metrics, retrieval, failures, cfg)
    selected = select_val_condition(gate_rows, metrics, cfg)
    best_auc = max(
        [metric_value(metrics, condition, CLCB) for condition in (E1, E2) if math.isfinite(metric_value(metrics, condition, CLCB))],
        default=float("nan"),
    )
    major_issues = failures[
        (failures["stage"] == "inner_dev") & (failures["major_issue"].astype(bool))
    ]
    if selected is not None:
        val_auc = metric_value(val_metrics, selected, CLCB, stage="official_val") if not val_metrics.empty else float("nan")
        val_strong = math.isfinite(val_auc) and val_auc >= cfg.strong_auc_threshold
        classification = "A. STRONG_EXTERNAL_FINGERPRINT_BASELINE" if val_strong else "B. PARTIAL_FINGERPRINT_SPECIFIC_SIGNAL"
        reason = (
            "Selected FLARE condition passed inner-dev and official VAL strong AUC gate."
            if val_strong
            else "Selected FLARE condition passed inner-dev, but official VAL did not satisfy the strong AUC gate."
        )
        official_val_gate = {
            "opened": True,
            "selected_condition": selected,
            "official_val_clcb_auc": val_auc,
            "strong_val_auc_threshold": cfg.strong_auc_threshold,
            "passed": bool(val_strong),
        }
    elif not major_issues.empty:
        classification = "D. INTEGRATION_INCONCLUSIVE"
        reason = "Coverage, extraction, score-join, or score-degeneracy failures prevent a valid FLARE conclusion."
        official_val_gate = {"opened": False, "reason": "Inner-dev integration gate failed due major issues."}
    elif math.isfinite(best_auc) and best_auc >= max(cfg.material_auc_threshold, EXISTING_CUSTOM_BASELINE_AUC + 0.03):
        classification = "B. PARTIAL_FINGERPRINT_SPECIFIC_SIGNAL"
        reason = "FLARE improved beyond the best existing custom inner-dev baseline but did not pass the strong gate."
        official_val_gate = {"opened": False, "reason": "Inner-dev strong gate failed."}
    else:
        classification = "C. EXTERNAL_BASELINE_ALSO_FAILS"
        reason = "FLARE stayed near chance or only weakly above the current baselines on inner-dev."
        official_val_gate = {"opened": False, "reason": "Inner-dev strong gate failed."}
    return {
        "classification": classification,
        "primary_reason": reason,
        "dependency_failures": dependency_failures,
        "inner_dev_gate": gate_rows,
        "selected_official_val_condition": val_condition or selected,
        "gate_thresholds": {
            "strong_auc_threshold": cfg.strong_auc_threshold,
            "material_auc_threshold": cfg.material_auc_threshold,
            "material_retrieval_mrr_gain": cfg.material_retrieval_mrr_gain,
            "within_same_session_min_auc": cfg.within_same_session_min_auc,
            "within_cross_session_min_auc": cfg.within_cross_session_min_auc,
            "max_failure_fraction": cfg.max_failure_fraction,
            "existing_custom_baseline_auc": EXISTING_CUSTOM_BASELINE_AUC,
            "baseline_retrieval": BASELINE_RETRIEVAL,
            "selection_rule": (
                "Open official VAL only for passing conditions; if both pass, select higher inner-dev CL->CB AUC, "
                "with RegressionPose chosen on an AUC tie within selection_auc_tie_margin."
            ),
        },
        "official_val_gate": official_val_gate,
        "test_gate": {"opened": False, "reason": "TEST remains prohibited for Phase 4C.1."},
    }


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [f for f in frames if f is not None and not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def run(
    *,
    manifest_dir: Path,
    phase4b1_dir: Path,
    flare_repo: Path,
    outdir: Path,
    polyu_root: Optional[str],
    gpu: str,
    cfg: FlareConfig,
    keep_staging: bool = False,
    allow_copy_staging: bool = False,
    p2_reference_csv: Path = resolve_repo_path(DEFAULT_P2_REFERENCE),
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(manifest_dir)
    phase4b1_dir = Path(phase4b1_dir)
    flare_repo = Path(flare_repo)

    provenance = flare_provenance(flare_repo)
    provenance["dependency_versions"] = dependency_versions()
    weights = model_weight_inventory(flare_repo)
    write_json(outdir / "upstream_provenance.json", provenance)
    write_csv(outdir / "model_weight_inventory.csv", weights)

    dependency_failures = validate_flare_dependency(flare_repo)
    before_hashes = canonical_hashes(manifest_dir)
    adapter_frames: list[pd.DataFrame] = []
    verification_rows: list[dict[str, Any]] = e0_reference_rows(p2_reference_csv)
    retrieval_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    output_inventory_frames: list[pd.DataFrame] = []
    stage_summaries: dict[str, Any] = {}
    val_condition: Optional[str] = None
    official_val_result: Optional[dict[str, Any]] = None

    if not dependency_failures:
        inner_data = load_stage_data(
            stage="inner_dev",
            manifest_dir=manifest_dir,
            phase4b1_dir=phase4b1_dir,
            polyu_root=polyu_root,
            cfg=cfg,
        )
        stage_summaries["inner_dev"] = {
            "split": inner_data.split,
            "identity_count": len(inner_data.identity_ids),
            "image_count": int(len(inner_data.images)),
            "pair_counts": inner_data.pair_counts,
            "polyu_root": {
                "path": str(inner_data.resolved_root.root) if inner_data.resolved_root.root else None,
                "source": inner_data.resolved_root.source,
                "exists": bool(inner_data.resolved_root.exists),
            },
        }
        for condition in (E1, E2):
            result = evaluate_condition(
                condition=condition,
                stage_data=inner_data,
                flare_repo=flare_repo,
                outdir=outdir,
                gpu=gpu,
                cfg=cfg,
                keep_staging=keep_staging,
                allow_copy_staging=allow_copy_staging,
            )
            adapter_frames.append(result["adapter"])
            verification_rows.extend(result["verification_rows"])
            retrieval_frames.append(result["retrieval"])
            runtime_rows.extend(result["runtime_rows"])
            failure_rows.extend(result["failure_rows"])
            output_inventory_frames.append(result["output_inventory"])

        metrics_now = pd.DataFrame(verification_rows)
        retrieval_now = merge_frames(retrieval_frames)
        failures_now = pd.DataFrame(failure_rows)
        gate_rows = condition_gate_rows(metrics_now, retrieval_now, failures_now, cfg)
        val_condition = select_val_condition(gate_rows, metrics_now, cfg)
        if val_condition is not None:
            val_data = load_stage_data(
                stage="official_val",
                manifest_dir=manifest_dir,
                phase4b1_dir=phase4b1_dir,
                polyu_root=polyu_root,
                cfg=cfg,
            )
            stage_summaries["official_val"] = {
                "split": val_data.split,
                "identity_count": len(val_data.identity_ids),
                "image_count": int(len(val_data.images)),
                "pair_counts": val_data.pair_counts,
                "opened_by_condition": val_condition,
            }
            official_val_result = evaluate_condition(
                condition=val_condition,
                stage_data=val_data,
                flare_repo=flare_repo,
                outdir=outdir,
                gpu=gpu,
                cfg=cfg,
                keep_staging=keep_staging,
                allow_copy_staging=allow_copy_staging,
            )
            adapter_frames.append(official_val_result["adapter"])
            verification_rows.extend(official_val_result["verification_rows"])
            retrieval_frames.append(official_val_result["retrieval"])
            runtime_rows.extend(official_val_result["runtime_rows"])
            failure_rows.extend(official_val_result["failure_rows"])
            output_inventory_frames.append(official_val_result["output_inventory"])

    adapter_manifest = merge_frames(adapter_frames)
    verification = pd.DataFrame(verification_rows)
    retrieval = merge_frames(retrieval_frames)
    runtime = pd.DataFrame(runtime_rows)
    failures = pd.DataFrame(failure_rows)
    output_inventory = merge_frames(output_inventory_frames)
    val_metrics = verification[verification["stage"] == "official_val"].copy() if not verification.empty else pd.DataFrame()
    decision = classify_decision(
        metrics=verification,
        retrieval=retrieval,
        failures=failures if not failures.empty else pd.DataFrame(columns=["stage", "major_issue", "condition"]),
        val_metrics=val_metrics,
        cfg=cfg,
        dependency_failures=dependency_failures,
        val_condition=val_condition,
    )

    write_csv(outdir / "adapter_manifest.csv", adapter_manifest)
    write_csv(outdir / "verification_metrics.csv", verification)
    write_csv(outdir / "retrieval_metrics.csv", retrieval)
    write_csv(outdir / "runtime_metrics.csv", runtime)
    write_csv(outdir / "failure_diagnostics.csv", failures)
    write_csv(outdir / "flare_output_inventory.csv", output_inventory)
    write_json(outdir / "inner_dev_gate_decision.json", decision)

    after_hashes = canonical_hashes(manifest_dir)
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "dataset": base.DATASET_NAME,
        "conditions": {
            E0: {
                "description": "Existing P2 SourceAFIS reference rows reused from prior diagnostics; not rerun in this phase.",
                "source": str(p2_reference_csv),
            },
            E1: {"pose": "RegressionPose", "descriptor": "FDD", "zero_shot": True},
            E2: {"pose": "VotingPose", "descriptor": "FDD", "zero_shot": True},
        },
        "stage_summaries": stage_summaries,
        "upstream": provenance,
        "model_weight_inventory": weights.to_dict("records"),
        "dependency_failures": dependency_failures,
        "decision": decision,
        "outputs": {
            "upstream_provenance_json": str(outdir / "upstream_provenance.json"),
            "model_weight_inventory_csv": str(outdir / "model_weight_inventory.csv"),
            "adapter_manifest_csv": str(outdir / "adapter_manifest.csv"),
            "verification_metrics_csv": str(outdir / "verification_metrics.csv"),
            "retrieval_metrics_csv": str(outdir / "retrieval_metrics.csv"),
            "runtime_metrics_csv": str(outdir / "runtime_metrics.csv"),
            "failure_diagnostics_csv": str(outdir / "failure_diagnostics.csv"),
            "inner_dev_gate_decision_json": str(outdir / "inner_dev_gate_decision.json"),
            "run_manifest_json": str(outdir / "run_manifest.json"),
        },
        "canonical_artifact_sha256_before": before_hashes,
        "canonical_artifact_sha256_after": after_hashes,
        "canonical_artifacts_unchanged": before_hashes == after_hashes,
        "canonical_artifacts_not_read": {
            "pairs_test_csv": str(Path(manifest_dir) / "pairs_test.csv"),
            "reason": "TEST remains closed; hash may be recorded for integrity only and pairs are not loaded for scoring.",
        },
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": dependency_versions(),
        "constraints": {
            "test_pairs_read_for_scoring": False,
            "test_images_loaded": False,
            "official_val_opened": bool(decision["official_val_gate"].get("opened", False)),
            "official_val_condition_frozen_before_evaluation": bool(val_condition is not None),
            "canonical_clcb_pair_bundle_regenerated": False,
            "canonical_manifest_or_pairs_modified": before_hashes == after_hashes,
            "raw_images_modified": False,
            "staging_uses_hardlinks_or_symlinks_where_supported": True,
            "permanent_duplicate_biometric_dataset_created": False,
            "flare_architecture_modified": False,
            "flare_model_finetuned": False,
            "used_p2_for_flare": False,
            "used_custom_clahe_or_enhancement_for_flare": False,
            "used_sourceafis_deep_sift_fusion": False,
            "implemented_ridgeformer": False,
            "additional_pose_methods": False,
            "parameter_grid_search": False,
        },
    }
    write_json(outdir / "run_manifest.json", run_manifest)
    return {
        "outdir": outdir,
        "provenance": provenance,
        "weights": weights,
        "verification": verification,
        "retrieval": retrieval,
        "runtime": runtime,
        "failures": failures,
        "decision": decision,
        "run_manifest": run_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4C.1 PolyU Cross FLARE external zero-shot baseline.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--phase4b1_dir", type=str, default=DEFAULT_PHASE4B1_DIR)
    parser.add_argument("--flare_repo", type=str, default=DEFAULT_FLARE_REPO)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--polyu_root", type=str, default="")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--p2_reference_csv", type=str, default=DEFAULT_P2_REFERENCE)
    parser.add_argument("--keep_staging", action="store_true")
    parser.add_argument("--allow_copy_staging", action="store_true")
    parser.add_argument("--poll_interval_seconds", type=float, default=FlareConfig.poll_interval_seconds)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = FlareConfig(poll_interval_seconds=float(args.poll_interval_seconds))
    try:
        result = run(
            manifest_dir=resolve_repo_path(args.data_dir),
            phase4b1_dir=resolve_repo_path(args.phase4b1_dir),
            flare_repo=resolve_repo_path(args.flare_repo),
            outdir=resolve_repo_path(args.outdir),
            polyu_root=str(args.polyu_root).strip() or None,
            gpu=str(args.gpu),
            cfg=cfg,
            keep_staging=bool(args.keep_staging),
            allow_copy_staging=bool(args.allow_copy_staging),
            p2_reference_csv=resolve_repo_path(args.p2_reference_csv),
        )
    except (FlareBaselineError, base.AlignmentError, local.LocalCorrespondenceError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    print("\n=== PolyU Cross Phase 4C.1 FLARE external baseline complete ===")
    print(f"Output dir       : {result['outdir']}")
    print(f"FLARE commit     : {result['provenance'].get('commit_sha')}")
    print(f"Classification   : {decision['classification']}")
    print(f"Official VAL     : {'opened' if decision['official_val_gate'].get('opened') else 'closed'}")
    print("TEST             : closed")
    if not result["verification"].empty:
        show = result["verification"][
            result["verification"]["condition"].isin([E1, E2])
        ][["condition", "stage", "protocol", "pair_count", "scored_count", "failed_count", "roc_auc", "eer"]]
        print("\nVerification:")
        print(show.to_string(index=False))
    if not result["retrieval"].empty:
        print("\nRetrieval:")
        print(result["retrieval"][["condition", "stage", "direction", "recall_at_1", "recall_at_5", "mrr"]].to_string(index=False))
    if not result["runtime"].empty:
        print("\nRuntime:")
        print(result["runtime"][["condition", "stage", "operation", "reported_seconds_per_image", "reported_seconds_per_pair", "peak_gpu_memory_mib"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
