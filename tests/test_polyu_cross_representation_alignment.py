from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.diagnostics import run_polyu_cross_representation_alignment as align


def test_inner_split_is_deterministic_and_identity_disjoint() -> None:
    ids = [str(i) for i in range(1, 101)]
    a = align.make_inner_split(ids, dev_fraction=0.15, seed=123)
    b = align.make_inner_split(list(reversed(ids)), dev_fraction=0.15, seed=123)
    assert a == b
    assert len(a["inner_dev"]) == 15
    assert len(a["inner_train"]) == 85
    assert not set(a["inner_train"]).intersection(a["inner_dev"])


def test_epoch_batches_are_identity_balanced_and_shuffled_control_rotates_keys() -> None:
    pools = {
        "1": {align.CONTACTLESS: ["cl1a", "cl1b"], align.CONTACT: ["cb1a", "cb1b"]},
        "2": {align.CONTACTLESS: ["cl2a", "cl2b"], align.CONTACT: ["cb2a", "cb2b"]},
        "3": {align.CONTACTLESS: ["cl3a", "cl3b"], align.CONTACT: ["cb3a", "cb3b"]},
        "4": {align.CONTACTLESS: ["cl4a", "cl4b"], align.CONTACT: ["cb4a", "cb4b"]},
    }
    normal = align.epoch_batches(pools, pools.keys(), batch_identities=4, epoch=1, seed=9)
    shuffled = align.epoch_batches(pools, pools.keys(), batch_identities=4, epoch=1, seed=9, shuffled_identity=True)

    assert len(normal) == 1
    batch = normal[0]
    assert len(batch["identity_ids"]) == 4
    assert len(batch["cl_uids"]) == 4
    assert len(batch["cb_uids"]) == 4
    assert batch["cb_identity_ids"] == batch["identity_ids"]

    shuf = shuffled[0]
    assert shuf["identity_ids"] != shuf["cb_identity_ids"]
    assert sorted(shuf["identity_ids"]) == sorted(shuf["cb_identity_ids"])


def test_symmetric_infonce_rewards_matching_diagonal() -> None:
    z = torch.eye(4)
    good = align.symmetric_infonce(z, z, temperature=0.07)
    bad = align.symmetric_infonce(z, torch.roll(z, shifts=1, dims=0), temperature=0.07)
    assert float(good) < 1e-4
    assert float(bad) > float(good) + 10.0


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(4, 4),
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _tiny_model() -> align.AlignmentModel:
    return align.AlignmentModel(TinyEncoder(), encoder_dim=4, projection_dim=2)


def test_trainability_policies_match_r1_r2_r3_contract() -> None:
    r1 = _tiny_model()
    names_r1 = align.configure_trainability(r1, "R1_projection_only")
    assert names_r1 == ["projection.weight", "projection.bias"]

    r2 = _tiny_model()
    names_r2 = set(align.configure_trainability(r2, "R2_partial_encoder_adaptation"))
    assert "encoder.net.3.weight" in names_r2
    assert "encoder.net.6.weight" in names_r2
    assert "projection.weight" in names_r2
    assert "encoder.net.0.weight" not in names_r2
    assert "encoder.net.1.weight" not in names_r2

    r3 = _tiny_model()
    names_r3 = set(align.configure_trainability(r3, "R3_full_shared_encoder_adaptation"))
    assert "encoder.net.0.weight" in names_r3
    assert "encoder.net.6.weight" in names_r3
    assert "projection.weight" in names_r3


def test_selection_rule_prefers_simplest_within_margin_and_no_degrade() -> None:
    rows = []
    protocols = [
        "contactless_to_contact_based",
        "contactless_to_contactless_same_session",
        "contactless_to_contactless_cross_session",
        "contact_based_to_contact_based_same_session",
        "contact_based_to_contact_based_cross_session",
    ]

    def add(condition: str, clcb_auc: float, within_auc: float) -> None:
        for protocol in protocols:
            rows.append(
                {
                    "condition": condition,
                    "protocol": protocol,
                    "roc_auc": clcb_auc if protocol == "contactless_to_contact_based" else within_auc,
                }
            )

    add("R0_zero_shot_embedding", 0.51, 0.72)
    add("R1_projection_only", 0.64, 0.71)
    add("R2_partial_encoder_adaptation", 0.655, 0.70)
    add("R3_full_shared_encoder_adaptation", 0.659, 0.71)

    train_results = {
        "R1_projection_only": SimpleNamespace(best_epoch=2),
        "R2_partial_encoder_adaptation": SimpleNamespace(best_epoch=3),
        "R3_full_shared_encoder_adaptation": SimpleNamespace(best_epoch=4),
    }
    decision = align.select_condition(rows, train_results)
    # R1 is within 0.02 of the best (R3), so the predeclared simplicity rule wins.
    assert decision["selected_condition"] == "R1_projection_only"
    assert decision["selected_epoch"] == 2


def test_pair_metrics_preserve_pair_count_and_stats() -> None:
    df = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3", "p4"],
            "label": [1, 1, 0, 0],
            "sample_uid_a": ["a1", "a2", "a3", "a4"],
            "sample_uid_b": ["b1", "b2", "b3", "b4"],
            "finger_unit_a": ["1", "2", "3", "4"],
            "finger_unit_b": ["1", "2", "5", "6"],
        }
    )
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    row = align.metric_row(condition="toy", stage="unit", protocol="p", df=df, scores=scores, epoch=1)
    assert row["pair_count"] == 4
    assert row["genuine_count"] == 2
    assert row["impostor_count"] == 2
    assert row["roc_auc"] == 1.0
    assert row["genuine_cosine_median"] == 0.8500000000000001
