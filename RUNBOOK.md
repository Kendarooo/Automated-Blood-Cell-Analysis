# Runbook — Pipeline BCCD

Flujo completo de comandos para reproducir todos los resultados del proyecto,
desde setup hasta evaluación final (FR-14).

---

## 0. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
wandb login          # pegar API key de https://wandb.ai/authorize
```

Verificar configuración antes de correr:
```bash
cat configs/config.yaml   # ajustar rutas si el entorno difiere
```

> **Nota:** la calibración encontró que `detection.conf_threshold = 0.4`
> maximiza F1. La configuración actual ya utiliza ese valor para reproducir
> los resultados documentados.

---

## 1. Tests

```bash
pytest tests/ -v
```

Módulos individuales:
```bash
pytest tests/test_transforms.py -v
pytest tests/test_extractor.py -v
pytest tests/test_ann_training.py -v
pytest tests/test_statistical_inference.py -v
```

---

## 2. Entrenar detector YOLO (FR-1 / FR-2)

```bash
python main.py --stage train_detector --config configs/config.yaml
```

Pesos guardados en: `runs/detect/outputs/yolo_finetune/weights/best.pt`

---

## 3. Calibrar umbral de confianza (FR-3 / evaluación)

```bash
python -m src.evaluation.calibrate_threshold \
    --config configs/config.yaml
```

Resultados en: `outputs/evaluation/calibration/summary.json`
(Mejor umbral encontrado: `0.4`, F1 = 0.7958)

---

## 4. Extraer features (FR-5 / FR-6)

```bash
python main.py --stage extract_features --config configs/config.yaml
```

Features normalizadas guardadas en: `outputs/features/`

---

## 5. Entrenar clasificadores (FR-8 / FR-9)

### ANN
```bash
python main.py --stage train_ann --config configs/config.yaml
```

### SVM
```bash
python main.py --stage train_svm --config configs/config.yaml
```

Artifacts persistidos en: `outputs/runs/<run_id>/`

### Con sweep W&B (FR-9 / FR-11)
```bash
# Crear sweep en W&B
wandb sweep configs/sweeps/ann.yaml
wandb agent <sweep_id>

wandb sweep configs/sweeps/svm.yaml
wandb agent <sweep_id>
```

O usando el CLI de experimentos directamente:
```bash
python -m src.experiments.cli --runner ann --config configs/config.yaml
python -m src.experiments.cli --runner svm --config configs/config.yaml
```

---

## 6. Construir baseline estadístico (FR-10)

```bash
python main.py --stage build_baseline \
    --config configs/config.yaml \
    --baseline outputs/inference/baseline.json
```

---

## 7. Inferencia sobre una imagen (FR-12 / FR-13)

```bash
python main.py --stage infer_image \
    --config configs/config.yaml \
    --image data/bccd/test/images/<imagen>.jpg \
    --classifier-run-dir outputs/runs/<run_id> \
    --baseline outputs/inference/baseline.json
```

O con el CLI de inferencia:
```bash
python -m src.inference.infer_cli \
    --image data/bccd/test/images/<imagen>.jpg \
    --classifier-run-dir outputs/runs/<run_id> \
    --config configs/config.yaml
```

---

## 8. Comparar etiquetas — evaluación final (FR-14)

```bash
python -m src.evaluation.compare_labels \
    --classifier-run-dir outputs/runs/<run_id> \
    --test-dir data/bccd/test \
    --output-dir outputs/evaluation/compare_labels \
    --iou-threshold 0.5 \
    --conf-threshold 0.4
```

Resultados en: `outputs/evaluation/compare_labels/summary.json`

---

## 9. Pipeline completo en un solo comando

Entrena todo el pipeline (detector → features → ANN → SVM → baseline).
Agrega `--image` para incluir inferencia al final.

```bash
python main.py --stage all --config configs/config.yaml

# Con inferencia incluida:
python main.py --stage all \
    --config configs/config.yaml \
    --image data/bccd/test/images/<imagen>.jpg
```

---

## Estructura de outputs esperada

```
outputs/
├── runs/<run_id>/          # artifacts ANN/SVM (modelo + normalizador)
├── features/               # features extraídas (.npz)
├── inference/
│   └── baseline.json       # proporciones de referencia desde train
└── evaluation/
    ├── calibration/
    │   └── summary.json    # mejor conf_threshold, precision/recall/F1
    └── compare_labels/
        ├── summary.json    # matched, FP, FN, matrices de confusión
        ├── per_image.csv
        └── confusion.csv
```
