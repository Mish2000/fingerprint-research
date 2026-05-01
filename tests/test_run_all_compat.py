import pytest

from scripts import run_all


def test_run_all_shim_routes_eval_all_to_run_benchmark_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_dispatch(target: str, argv: list[str]) -> int:
        captured["target"] = target
        captured["argv"] = argv
        return 11

    monkeypatch.setattr(run_all, "dispatch_legacy_args", fake_dispatch)

    assert run_all.main(["--eval_all"]) == 11
    assert captured["target"] == run_all.BATCH_TARGET
    assert "--methods" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--methods") + 1] == "classic_v2,harris,sift,dl_quick,dedicated,vit"


def test_run_all_shim_routes_eval_one_to_run_benchmark_once(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_dispatch(target: str, argv: list[str]) -> int:
        captured["target"] = target
        captured["argv"] = argv
        return 12

    monkeypatch.setattr(run_all, "dispatch_legacy_args", fake_dispatch)

    assert run_all.main(["--eval_one", "--method", "vit", "--split", "test"]) == 12
    assert captured["target"] == run_all.ONCE_TARGET
    assert captured["argv"][captured["argv"].index("--method") + 1] == "vit"
    assert captured["argv"][captured["argv"].index("--split") + 1] == "test"


def test_run_all_shim_defaults_to_batch_runner_when_no_mode_flag_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_dispatch(target: str, argv: list[str]) -> int:
        captured["target"] = target
        captured["argv"] = argv
        return 13

    monkeypatch.setattr(run_all, "dispatch_legacy_args", fake_dispatch)

    assert run_all.main([]) == 13
    assert captured["target"] == run_all.BATCH_TARGET
    assert "--splits" in captured["argv"]


def test_run_all_shim_rejects_both_eval_all_and_eval_one() -> None:
    with pytest.raises(SystemExit):
        run_all.main(["--eval_all", "--eval_one"])
