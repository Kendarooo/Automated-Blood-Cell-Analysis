"""Tests for the FR-11 CLI experiment entrypoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.experiments import cli as cli_module


def test_cli_dispatches_ann_runner(monkeypatch, capsys) -> None:
    """ANN mode must prepare features once and dispatch to the ANN runner."""
    captured: dict[str, object] = {}
    prepared = SimpleNamespace()

    def fake_load_config(path: str):
        captured["config_path"] = path
        return {"seed": 7, "wandb": {"enabled": True}}

    def fake_set_seed(seed: int):
        captured["seed"] = seed

    def fake_apply_wandb_overrides(cfg):
        captured["wandb_cfg"] = cfg
        return cfg, False

    def fake_prepare_features(cfg):
        captured["prepared_cfg"] = cfg
        return prepared

    def fake_run_ann(cfg, incoming_prepared):
        captured["runner"] = "ann"
        captured["runner_cfg"] = cfg
        captured["prepared"] = incoming_prepared
        return {"run_id": "ann_1234", "val_macro_f1": 0.75}

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "_apply_wandb_overrides", fake_apply_wandb_overrides)
    monkeypatch.setattr(cli_module, "set_seed", fake_set_seed)
    monkeypatch.setattr(cli_module, "_prepare_features", fake_prepare_features)
    monkeypatch.setattr(cli_module, "run_ann_experiment", fake_run_ann)

    result = cli_module.main(["--runner", "ann", "--config", "custom.yaml"])
    stdout = capsys.readouterr().out

    assert result["run_id"] == "ann_1234"
    assert captured["config_path"] == "custom.yaml"
    assert captured["seed"] == 7
    assert captured["runner"] == "ann"
    assert captured["prepared"] is prepared
    assert json.loads(stdout)["val_macro_f1"] == 0.75


def test_cli_dispatches_svm_runner_and_can_disable_wandb(
    monkeypatch,
    capsys,
) -> None:
    """SVM mode must honor --disable-wandb before dispatch."""
    captured: dict[str, object] = {}
    prepared = SimpleNamespace()

    def fake_load_config(path: str):
        del path
        return {"seed": 42, "wandb": {"enabled": True}}

    def fake_apply_wandb_overrides(cfg):
        captured["wandb_enabled_before_prepare"] = cfg["wandb"]["enabled"]
        return cfg, False

    def fake_prepare_features(cfg):
        captured["wandb_enabled"] = cfg["wandb"]["enabled"]
        return prepared

    def fake_run_svm(cfg, incoming_prepared):
        captured["runner"] = "svm"
        captured["runner_cfg"] = cfg
        captured["prepared"] = incoming_prepared
        return {"run_id": "svm_5678", "val_macro_f1": 0.61}

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "_apply_wandb_overrides", fake_apply_wandb_overrides)
    monkeypatch.setattr(cli_module, "set_seed", lambda seed: captured.setdefault("seed", seed))
    monkeypatch.setattr(cli_module, "_prepare_features", fake_prepare_features)
    monkeypatch.setattr(cli_module, "run_svm_experiment", fake_run_svm)

    result = cli_module.main(["--runner", "svm", "--disable-wandb"])
    stdout = capsys.readouterr().out

    assert result["run_id"] == "svm_5678"
    assert captured["runner"] == "svm"
    assert captured["prepared"] is prepared
    assert captured["wandb_enabled"] is False
    assert json.loads(stdout)["run_id"] == "svm_5678"


def test_cli_merges_flattened_wandb_overrides() -> None:
    """Sweep overrides must replace nested config values before dispatch."""
    cfg = {
        "ann": {"lr": 0.001},
        "dataset": {"batch_size": 32},
        "wandb": {"enabled": True},
    }

    merged = cli_module._merge_flattened_overrides(
        cfg,
        {
            "ann.lr": 0.01,
            "dataset.batch_size": 64,
            "svm.gamma": "scale",
        },
    )

    assert merged["ann"]["lr"] == 0.01
    assert merged["dataset"]["batch_size"] == 64
    assert merged["svm"]["gamma"] == "scale"


def test_cli_rejects_invalid_runner() -> None:
    """Invalid runner names must fail clearly at argument parsing time."""
    with pytest.raises(SystemExit):
        cli_module.main(["--runner", "knn"])
