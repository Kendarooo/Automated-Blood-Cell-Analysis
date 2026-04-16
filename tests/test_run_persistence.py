"""Tests for local experiment run persistence in FR-11."""

from __future__ import annotations

import json
from pathlib import Path

from src.experiments.run_persistence import build_run_id, persist_run


def test_build_run_id_uses_classifier_prefix() -> None:
    """Run ids must be prefixed by classifier name."""
    run_id = build_run_id("ann")
    assert run_id.startswith("ann_")
    assert len(run_id) > 4


def test_persist_run_writes_expected_json_files(tmp_path: Path) -> None:
    """Local persistence must create summary, config, metrics, and metadata JSONs."""
    model_path = tmp_path / "external_model.pt"
    model_path.write_text("model", encoding="utf-8")

    result = persist_run(
        run_id="ann_1234abcd",
        classifier_name="ann",
        output_root=tmp_path,
        effective_config={"ann": {"lr": 1e-3}},
        metrics={"val_macro_f1": 0.75, "val_accuracy": 0.8},
        feature_metadata={"feature_dim": 256, "class_names": ["WBC", "RBC", "Platelets"]},
        model_path=model_path,
    )

    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "effective_config.json").exists()
    assert (output_dir / "feature_metadata.json").exists()
    assert (output_dir / "run_summary.json").exists()

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "ann_1234abcd"
    assert summary["classifier_name"] == "ann"
    assert summary["metric_name"] == "val_macro_f1"
    assert summary["metric_value"] == 0.75
    assert summary["local_artifact_paths"]["model"] == str(model_path)
