"""Baseline estimation and persistence for FR-10 statistical inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineDistribution:
    """Empirical baseline distribution estimated from the training split only."""

    class_order: list[str]
    proportions: dict[str, float]
    config_fingerprint: str
    created_from_split: str


class ConfigFingerprint:
    """Utility class to build stable fingerprints from configuration dictionaries."""

    @staticmethod
    def build(config: dict) -> str:
        """Return a stable SHA-256 fingerprint for a configuration dictionary."""
        canonical_json = json.dumps(
            ConfigFingerprint._select_relevant_config(config),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _select_relevant_config(config: dict) -> dict:
        """Keep only config fields that materially affect baseline validity."""
        dataset_cfg = config.get("dataset", {})
        inference_cfg = config.get("inference", {})

        if not dataset_cfg and not inference_cfg:
            return config

        return {
            "dataset": {
                "train_dir": dataset_cfg.get("train_dir"),
                "class_names": dataset_cfg.get("class_names"),
            },
            "inference": inference_cfg,
        }


class BaselineEstimator:
    """
    Estimate class proportions using ONLY the training split.

    This class must never consume validation, test, or synthetic data,
    in order to preserve NFR-4 and avoid leakage.
    """

    def __init__(self, class_order: list[str]) -> None:
        self._class_order = class_order

    def fit_from_train_counts(
        self,
        train_counts: list[dict[str, int]],
        config: dict,
    ) -> BaselineDistribution:
        """
        Estimate the baseline distribution from train-set counts only.

        Args:
            train_counts: Per-image class counts from the training split only.
            config: Configuration used to generate the baseline.

        Returns:
            BaselineDistribution with empirical proportions.
        """
        aggregated_counts = {
            class_name: 0
            for class_name in self._class_order
        }

        for counts in train_counts:
            for class_name in self._class_order:
                aggregated_counts[class_name] += int(counts.get(class_name, 0))

        total_cells = sum(aggregated_counts.values())
        if total_cells == 0:
            raise ValueError(
                "Cannot estimate baseline distribution: no training cells were provided."
            )

        proportions = {
            class_name: aggregated_counts[class_name] / total_cells
            for class_name in self._class_order
        }

        return BaselineDistribution(
            class_order=self._class_order,
            proportions=proportions,
            config_fingerprint=ConfigFingerprint.build(config),
            created_from_split="train",
        )


class BaselineRepository:
    """Persist and load baseline distributions from disk."""

    @staticmethod
    def save(baseline: BaselineDistribution, path: str) -> None:
        """Save a baseline distribution as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(asdict(baseline), handle, indent=2)

    @staticmethod
    def load(path: str, expected_config: dict | None = None) -> BaselineDistribution:
        """
        Load a baseline distribution and optionally validate its config fingerprint.

        Args:
            path: Path to the saved baseline JSON file.
            expected_config: Optional current config to validate against.

        Returns:
            Loaded BaselineDistribution.
        """
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        baseline = BaselineDistribution(**payload)

        if expected_config is not None:
            expected_fingerprint = ConfigFingerprint.build(expected_config)
            if baseline.config_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Loaded baseline distribution does not match the current configuration."
                )

        return baseline
