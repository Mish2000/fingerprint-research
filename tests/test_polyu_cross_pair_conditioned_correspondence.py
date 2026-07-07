from __future__ import annotations

import math

import pandas as pd
import torch
import torch.nn.functional as F

from scripts.diagnostics import run_polyu_cross_pair_conditioned_correspondence as pc


def test_2d_position_encoding_is_deterministic_and_expected_shape() -> None:
    first = pc.make_2d_sincos_position_encoding(24, 24, 128)
    second = pc.make_2d_sincos_position_encoding(24, 24, 128)
    assert first.shape == (576, 128)
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


def test_c1_and_c2_trainable_surfaces_match_phase_contract() -> None:
    c1 = pc.LocalCorrespondenceModel(condition=pc.C1, projection_dim=128, attention_heads=4)
    c2 = pc.LocalCorrespondenceModel(condition=pc.C2, projection_dim=128, attention_heads=4)

    c1_trainable, c1_total = pc.count_params(c1)
    c2_trainable, c2_total = pc.count_params(c2)
    assert c1_trainable == c1_total == 32896
    assert c2_trainable == c2_total == 165504
    assert set(pc.trainable_names(c1)) == {"projection.weight", "projection.bias"}
    assert c2.block is not None
    assert c2.block.self_attn.num_heads == 4
    assert c2.block.cross_attn.num_heads == 4
    assert not any(name.startswith("encoder.") for name in pc.trainable_names(c2))


def test_pair_conditioning_forward_returns_normalized_descriptors_and_attention_diagnostics() -> None:
    torch.manual_seed(0)
    model = pc.LocalCorrespondenceModel(condition=pc.C2, projection_dim=128, attention_heads=4)
    a = F.normalize(torch.randn(2, 576, 256), p=2, dim=2)
    b = F.normalize(torch.randn(2, 576, 256), p=2, dim=2)
    za, zb, diag = model.forward_pair(a, b, need_diagnostics=True)

    assert za.shape == (2, 576, 128)
    assert zb.shape == (2, 576, 128)
    assert torch.allclose(torch.linalg.vector_norm(za, dim=2), torch.ones(2, 576), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(zb, dim=2), torch.ones(2, 576), atol=1e-5)
    assert torch.isfinite(diag["self_attention_entropy"]).all()
    assert torch.isfinite(diag["cross_attention_entropy"]).all()
    assert ((diag["self_attention_entropy"] >= 0.0) & (diag["self_attention_entropy"] <= 1.0)).all()
    assert ((diag["cross_attention_entropy"] >= 0.0) & (diag["cross_attention_entropy"] <= 1.0)).all()


def test_sinkhorn_scores_and_diagnostics_are_differentiable_and_finite() -> None:
    a = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    b = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    cfg = pc.PairConditionedConfig()
    score, diag = pc.sinkhorn_scores_and_diagnostics(a, b, cfg=cfg, return_diagnostics=True)

    assert score.shape == (1,)
    assert float(score.item()) > 0.99
    for value in diag.values():
        assert torch.isfinite(value).all()
    score.sum().backward()
    assert a.grad is not None
    assert float(a.grad.abs().sum()) > 0.0


def test_candidate_scores_use_one_positive_then_deterministic_negatives() -> None:
    class ToyModel(torch.nn.Module):
        condition = pc.C1

        def forward_pair(self, a, b, need_diagnostics=False):
            return a, b, {}

    cfg = pc.PairConditionedConfig(negatives_per_anchor=2, train_pair_chunk_size=8)
    desc = F.normalize(torch.eye(4, dtype=torch.float32).reshape(4, 1, 4), p=2, dim=2)
    scores = pc.candidate_scores_for_direction(model=ToyModel(), anchor_desc=desc, candidate_desc=desc, cfg=cfg)

    assert scores.shape == (4, 3)
    assert torch.allclose(scores[:, 0], torch.ones(4), atol=1e-5)
    assert torch.all(scores[:, 1:] < 0.5)


def _result(condition: str, auc: float, cl_mrr: float, cb_mrr: float, within_auc: float) -> pc.TrainRunResult:
    metric_rows = [{"protocol": pc.CLCB, "roc_auc": auc}]
    retrieval_rows = [
        {"direction": "CL_probe_to_CB_gallery", "mrr": cl_mrr},
        {"direction": "CB_probe_to_CL_gallery", "mrr": cb_mrr},
    ]
    within_rows = [{"protocol": p, "roc_auc": within_auc} for p in pc.WITHIN_PROTOCOLS]
    attention_rows = [
        {
            "output_descriptor_variance_mean": 0.01,
            "near_identical_context_descriptor_fraction_mean": 0.0,
        }
    ]
    return pc.TrainRunResult(
        condition=condition,
        stage="unit",
        seed=13,
        best_epoch=1,
        best_auc=auc,
        best_state_dict={},
        curve_rows=[],
        metric_rows=metric_rows,
        retrieval_rows=retrieval_rows,
        within_rows=within_rows,
        correspondence_rows=[],
        attention_rows=attention_rows,
        trainable_param_count=1,
        total_param_count=1,
        trainable_names=["projection.weight"],
        gradient_check={"gradients_only_in_intended_components": True},
        encoder_sha256_before="same",
        encoder_sha256_after="same",
    )


def test_signal_gate_requires_auc_gain_retrieval_collapse_and_within_controls() -> None:
    cfg = pc.PairConditionedConfig()
    c0 = _result(pc.C0, 0.484, 0.11, 0.10, 0.80)
    good = _result(pc.C2, 0.63, 0.15, 0.12, 0.75)
    gate = pc.signal_gate_for_result(good, c0=c0, cfg=cfg)
    assert gate["passed"] is True

    low_auc = _result(pc.C2, 0.60, 0.15, 0.12, 0.75)
    assert pc.signal_gate_for_result(low_auc, c0=c0, cfg=cfg)["passed"] is False

    weak_retrieval = _result(pc.C2, 0.63, 0.10, 0.12, 0.75)
    assert pc.signal_gate_for_result(weak_retrieval, c0=c0, cfg=cfg)["passed"] is False

    collapsed = _result(pc.C2, 0.63, 0.15, 0.12, 0.75)
    collapsed.attention_rows[0]["near_identical_context_descriptor_fraction_mean"] = 0.5
    assert pc.signal_gate_for_result(collapsed, c0=c0, cfg=cfg)["passed"] is False

    within_collapse = _result(pc.C2, 0.63, 0.15, 0.12, 0.50)
    assert pc.signal_gate_for_result(within_collapse, c0=c0, cfg=cfg)["passed"] is False


def test_decision_d_when_stage2_conditions_fail_signal_gate() -> None:
    cfg = pc.PairConditionedConfig()
    c0 = _result(pc.C0, 0.484, 0.11, 0.10, 0.80)
    c1 = _result(pc.C1, 0.55, 0.10, 0.09, 0.75)
    c2 = _result(pc.C2, 0.56, 0.10, 0.09, 0.75)
    decision = pc.classify_decision(stage2_results=[c1, c2], stage3_results=[], c0=c0, cfg=cfg)
    assert decision["classification"] == "D. PAIR_CONDITIONED_LOCAL_ALIGNMENT_INSUFFICIENT"
    assert decision["stage3_multiseed"]["ran"] is False
    assert decision["official_val_gate"]["opened"] is False


def test_experiment_config_declares_exact_single_block_attention_and_sinkhorn() -> None:
    cfg = pc.PairConditionedConfig()
    config = pc.experiment_config(
        cfg,
        checkpoint=pc.resolve_repo_path(pc.DEFAULT_CHECKPOINT),
        phase4b1_dir=pc.resolve_repo_path(pc.DEFAULT_PHASE4B1_DIR),
    )
    assert config["attention"] == {"model_dim": 128, "heads": 4, "blocks": 1}
    assert config["sinkhorn"]["entropy_regularization"] == 0.07
    assert config["sinkhorn"]["iterations"] == 20
    assert config["training_candidates"]["batch_identities"] == 8
    assert config["training_candidates"]["negatives_per_anchor"] == 3
    assert math.isclose(config["training_candidates"]["temperature"], 0.07)
