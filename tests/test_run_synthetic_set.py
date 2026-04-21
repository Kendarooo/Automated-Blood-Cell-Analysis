"""Tests for batch inference over a synthetic FR-14 image directory."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.inference.run_synthetic_set import main


class StubPipeline:
    """Pipeline stub returning deterministic proportions and alert decisions."""

    def __init__(self, cfg: dict, artifact_paths) -> None:  # noqa: ANN001
        del cfg, artifact_paths

    def predict(self, image_path: Path):
        if image_path.stem == "alert_case":
            proportions = {"Platelets": 0.05, "RBC": 0.20, "WBC": 0.75}
            p_value = 0.001
            alert = True
        else:
            proportions = {"Platelets": 0.05, "RBC": 0.90, "WBC": 0.05}
            p_value = 0.80
            alert = False

        return SimpleNamespace(
            image_path=image_path,
            classifier_name="ann",
            num_detections=20,
            statistics=SimpleNamespace(proportions=proportions),
            statistical_test=SimpleNamespace(
                p_value=p_value,
                alert=alert,
                test_executed=True,
            ),
        )


def test_run_synthetic_set_writes_per_image_csv(monkeypatch, tmp_path: Path) -> None:
    """Batch synthetic inference must export one CSV row per image with alerts and p-values."""
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(synthetic_dir / "normal_case.jpg")
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(synthetic_dir / "alert_case.png")

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "effective_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "normalizer.npz").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "src.inference.run_synthetic_set.load_config",
        lambda _: {"detection": {"best_weights": "detector.pt"}},
    )
    monkeypatch.setattr(
        "src.inference.run_synthetic_set.InferencePipeline",
        StubPipeline,
    )

    summary = main(
        [
            "--synthetic-dir",
            str(synthetic_dir),
            "--classifier-run-dir",
            str(run_dir),
            "--baseline",
            str(baseline_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert summary["num_images"] == 2
    assert summary["num_alerts"] == 1

    csv_path = Path(summary["results_csv"])
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    alert_row = next(row for row in rows if row["image_path"].endswith("alert_case.png"))
    normal_row = next(row for row in rows if row["image_path"].endswith("normal_case.jpg"))

    assert alert_row["alert"] == "True"
    assert float(alert_row["wbc_pct"]) == 75.0
    assert float(alert_row["rbc_pct"]) == 20.0
    assert float(alert_row["plt_pct"]) == 5.0
    assert normal_row["alert"] == "False"
