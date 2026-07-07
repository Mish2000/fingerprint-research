from __future__ import annotations

import torch

from scripts.deep.score_fast_pair_ddp_splits import PairModel
from scripts.diagnostics import run_polyu_cross_learned_local_alignment as learned


def test_topk_local_pair_score_is_symmetric_and_differentiable() -> None:
    a = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    b = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    score, ab, ba = learned.local_pair_scores_from_descriptors(a, b, topk_fraction=0.5)
    assert score.shape == (1,)
    assert torch.allclose(score, torch.ones_like(score))
    assert torch.allclose(ab, torch.ones_like(ab))
    assert torch.allclose(ba, torch.ones_like(ba))
    score.sum().backward()
    assert a.grad is not None
    assert float(a.grad.abs().sum()) > 0.0


def test_pair_score_matrix_rewards_diagonal_matches() -> None:
    desc = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(3, 1, 1)
    desc[1] = torch.roll(desc[1], shifts=1, dims=0)
    desc[2] = torch.roll(desc[2], shifts=2, dims=0)
    scores = learned.pair_score_matrix(desc, desc, topk_fraction=0.25, pair_chunk=2)
    assert scores.shape == (3, 3)
    assert torch.allclose(torch.diag(scores), torch.ones(3))


def test_projection_trainability_policies_freeze_encoder() -> None:
    pair_model = PairModel(width=32, embedding_dim=512, hidden_dim=768)
    a1 = learned.LocalProjectionModel(pair_model, condition=learned.A1, projection_dim=128)
    names = {name for name, param in a1.named_parameters() if param.requires_grad}
    assert names == {"shared_projection.weight", "shared_projection.bias"}
    assert not any(name.startswith("encoder.") for name in names)

    pair_model = PairModel(width=32, embedding_dim=512, hidden_dim=768)
    a2 = learned.LocalProjectionModel(pair_model, condition=learned.A2, projection_dim=128)
    names = {name for name, param in a2.named_parameters() if param.requires_grad}
    assert names == {
        "cl_projection.weight",
        "cl_projection.bias",
        "cb_projection.weight",
        "cb_projection.bias",
    }
    assert not any(name.startswith("encoder.") for name in names)


def _result(condition: str, auc: float, mrr_gain: float = 0.04, collapse: bool = False) -> learned.LearnedLocalResult:
    retrieval = [
        {
            "direction": "CL_probe_to_CB_gallery",
            "recall_at_1": 0.10,
            "recall_at_5": 0.20,
            "mrr": 0.10 + mrr_gain,
        },
        {
            "direction": "CB_probe_to_CL_gallery",
            "recall_at_1": 0.10,
            "recall_at_5": 0.20,
            "mrr": 0.10 + mrr_gain,
        },
    ]
    return learned.LearnedLocalResult(
        condition=condition,
        seed=13,
        stage="unit",
        best_epoch=1,
        best_auc=auc,
        best_state_dict={},
        curve_rows=[],
        metric_rows=[{"roc_auc": auc}],
        retrieval_rows=retrieval,
        within_rows=[{"roc_auc": 0.80}],
        local_diag_rows=[],
        collapse_rows=[
            {
                "per_dim_std_mean": 0.0 if collapse else 0.01,
                "near_identical_descriptor_fraction_sampled": 1.0 if collapse else 0.0,
            }
        ],
        trainable_param_count=1,
        total_param_count=2,
        trainable_names=["head.weight"],
        gradient_check={"gradients_only_in_intended_components": True},
        encoder_sha256_before="same",
        encoder_sha256_after="same",
    )


def test_signal_gate_requires_auc_retrieval_no_collapse_and_gradients() -> None:
    cfg = learned.LearnedLocalConfig()
    baseline = _result(learned.A0, learned.BASELINE_F_RAW_L2_AUC, mrr_gain=0.0)
    good = _result(learned.A1, 0.62)
    gate = learned.signal_gate_for_result(good, baseline, cfg)
    assert gate["passed"] is True

    low_auc = _result(learned.A1, 0.56)
    assert learned.signal_gate_for_result(low_auc, baseline, cfg)["passed"] is False

    collapsed = _result(learned.A1, 0.62, collapse=True)
    assert learned.signal_gate_for_result(collapsed, baseline, cfg)["passed"] is False
