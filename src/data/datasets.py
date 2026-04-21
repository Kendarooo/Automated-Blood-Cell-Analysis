"""Dataset loaders for BCCD splits stored in YOLO detection format."""
# Author: Kendall Madrigal, Alexandra Alfaro / Claude Sonnet 4.6

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.transforms import TrainTransforms, ValTransforms


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

"""Estructura inmutable para almacenar las coordenadas absolutas en píxeles de las cajas delimitadoras."""
@dataclass(frozen=True)
class YoloAnnotation:
    """One YOLO-format annotation already converted to absolute pixels."""

    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

"""Inicializa el cargador y aplica las transformaciones de imagen correspondientes 
según si es el set de entrenamiento o validación."""
class BCCDYoloSplitLoader:
    """Load BCCD train/val splits by cropping annotated cells in memory."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._train_transform = TrainTransforms(cfg)
        self._val_transform = ValTransforms(cfg)
        self._train_dir = Path(cfg.get("dataset", {}).get("train_dir", "data/bccd/train"))

    def _get_transform(self, split_dir: Path) -> TrainTransforms | ValTransforms:
        """Return TrainTransforms for the train split, ValTransforms otherwise."""
        return (
            self._train_transform
            if Path(split_dir) == self._train_dir
            else self._val_transform
        )
    """Carga un split completo en memoria, recortando y transformando todas las 
    células anotadas para retornar tensores listos."""
    def load_split(
        self,
        split_dir: Path,
        class_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return crop tensors and encoded labels for one BCCD split."""
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(
                f"Expected YOLO split directories '{images_dir}' and '{labels_dir}'."
            )

        transform = self._get_transform(split_dir)
        crops: list[torch.Tensor] = []
        labels: list[int] = []

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = self._find_matching_image(images_dir, label_path.stem)
            if image_path is None:
                raise ValueError(
                    f"Could not find image matching label file '{label_path.name}'."
                )

            with Image.open(image_path).convert("RGB") as image:
                annotations = self._parse_annotations(
                    label_path=label_path,
                    image_width=image.width,
                    image_height=image.height,
                    num_classes=len(class_names),
                )

                for annotation in annotations:
                    crop = image.crop(
                        (annotation.x1, annotation.y1, annotation.x2, annotation.y2)
                    )
                    crops.append(transform(crop))
                    labels.append(annotation.class_id)

        if not crops:
            raise ValueError(f"No annotated cells found in split directory '{split_dir}'.")

        return torch.stack(crops), np.asarray(labels, dtype=np.int64)
    """Generador que recorta y procesa las imágenes en lotes (batches) para evitar la saturación de la memoria RAM."""
    def iter_split_batches(
        self,
        split_dir: Path,
        class_names: tuple[str, ...],
        batch_size: int,
    ):
        """Yield one batch of cropped cells at a time to limit peak memory usage."""
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(
                f"Expected YOLO split directories '{images_dir}' and '{labels_dir}'."
            )

        transform = self._get_transform(split_dir)
        batch_crops: list[torch.Tensor] = []
        batch_labels: list[int] = []
        yielded_any = False

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = self._find_matching_image(images_dir, label_path.stem)
            if image_path is None:
                raise ValueError(
                    f"Could not find image matching label file '{label_path.name}'."
                )

            with Image.open(image_path).convert("RGB") as image:
                annotations = self._parse_annotations(
                    label_path=label_path,
                    image_width=image.width,
                    image_height=image.height,
                    num_classes=len(class_names),
                )

                for annotation in annotations:
                    crop = image.crop(
                        (annotation.x1, annotation.y1, annotation.x2, annotation.y2)
                    )
                    batch_crops.append(transform(crop))
                    batch_labels.append(annotation.class_id)

                    if len(batch_crops) >= batch_size:
                        yielded_any = True
                        yield torch.stack(batch_crops), np.asarray(batch_labels, dtype=np.int64)
                        batch_crops = []
                        batch_labels = []

        if batch_crops:
            yielded_any = True
            yield torch.stack(batch_crops), np.asarray(batch_labels, dtype=np.int64)

        if not yielded_any:
            raise ValueError(f"No annotated cells found in split directory '{split_dir}'.")
    """Busca y retorna la ruta del archivo de imagen que corresponde exactamente al nombre del archivo de anotación."""
    @staticmethod
    def _find_matching_image(images_dir: Path, stem: str) -> Path | None:
        for suffix in SUPPORTED_IMAGE_SUFFIXES:
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None
    """Lee los archivos de texto YOLO, valida sus datos y convierte las coordenadas 
    relativas (centro, ancho, alto) a píxeles absolutos."""
    @staticmethod
    def _parse_annotations(
        label_path: Path,
        image_width: int,
        image_height: int,
        num_classes: int,
    ) -> list[YoloAnnotation]:
        annotations: list[YoloAnnotation] = []

        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation at {label_path}:{line_number}. "
                    "Expected 5 whitespace-separated values."
                )

            class_id = int(parts[0])
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(
                    f"Invalid class_id={class_id} at {label_path}:{line_number}. "
                    f"Expected range [0, {num_classes - 1}]."
                )

            x_center, y_center, box_width, box_height = map(float, parts[1:])
            x1, y1, x2, y2 = BCCDYoloSplitLoader._to_pixel_box(
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height,
            )
            annotations.append(
                YoloAnnotation(
                    class_id=class_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return annotations

    @staticmethod
    def _to_pixel_box(
        *,
        x_center: float,
        y_center: float,
        box_width: float,
        box_height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x1 = int(round((x_center - box_width / 2.0) * image_width))
        y1 = int(round((y_center - box_height / 2.0) * image_height))
        x2 = int(round((x_center + box_width / 2.0) * image_width))
        y2 = int(round((y_center + box_height / 2.0) * image_height))

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))
        return x1, y1, x2, y2
