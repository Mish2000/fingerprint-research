from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "week06+07+07"))  # so we can import train_patch_descriptor.py directly

import argparse
import numpy as np
import torch

from pipelines.benchmark.train_patch_descriptor import PatchPairStream, SimCLRModel, checkpoint_model_config  # <-- robust import


def load_sanity_checkpoint(ckpt_path: str | Path) -> tuple[dict, dict]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Descriptor checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format. Expected a dict checkpoint.")
    model_config = checkpoint_model_config(ckpt)
    return ckpt, model_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--manifest", type=str, default="data/processed/nist_sd300b/manifest.csv")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit_images", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)  # total evaluated pairs = steps*batch
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt, model_config = load_sanity_checkpoint(args.ckpt)
    train_args = ckpt.get("args", {})
    patch = int(model_config.get("patch", train_args.get("patch", 48)))
    max_kpts = int(train_args.get("max_kpts", 800))
    patches_per_image = int(train_args.get("patches_per_image", 32))

    model = SimCLRModel(
        emb_dim=int(model_config["emb_dim"]),
        proj_dim=int(model_config["proj_dim"]),
        model_arch=str(model_config["model_arch"]),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(
        "[SANITY] "
        f"model_arch={model_config['model_arch']} "
        f"emb_dim={model_config['emb_dim']} "
        f"proj_dim={model_config['proj_dim']} "
        f"patch={patch}"
    )

    ds = PatchPairStream(
        manifest_csv=args.manifest,
        split=args.split,
        patch=patch,
        max_kpts=max_kpts,
        patches_per_image=patches_per_image,
        seed=args.seed,
        limit_images=args.limit_images,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=0)
    it = iter(loader)

    pos_sims = []
    neg_sims = []

    with torch.no_grad():
        for _ in range(args.steps):
            x1, x2 = next(it)
            x1 = x1.to(device)
            x2 = x2.to(device)

            h1, _ = model(x1)
            h2, _ = model(x2)

            # positives (same patch different aug)
            pos = (h1 * h2).sum(dim=1)  # cosine since normalized
            pos_sims.append(pos.cpu().numpy())

            # negatives (shuffle within batch)
            idx = torch.randperm(h2.size(0), device=device)
            neg = (h1 * h2[idx]).sum(dim=1)
            neg_sims.append(neg.cpu().numpy())

    pos = np.concatenate(pos_sims)
    neg = np.concatenate(neg_sims)

    print(f"[SANITY] N={len(pos)}")
    print(f"[SANITY] pos_mean={pos.mean():.4f}  pos_std={pos.std():.4f}")
    print(f"[SANITY] neg_mean={neg.mean():.4f}  neg_std={neg.std():.4f}")
    print(f"[SANITY] gap(pos-neg)={pos.mean()-neg.mean():.4f}")

    good = (pos.mean() > neg.mean() + 0.10)
    print(f"[SANITY] PASS={good}")


if __name__ == "__main__":
    main()
