"""
=============================================================================
Autor original: Gemini 3.1 Pro (Bajo la dirección de Pablo Alvarado Moya)
Curso: EL 5857 Aprendizaje Automático
Descripción: Script modular para la descarga, verificación y exploración 
             interactiva del conjunto de datos BCCD.
Auditoría de IA: Este archivo fue generado en su totalidad por un LLM 
                 (vibe coding) con modificaciones manuales mínimas para 
                 ajustar las rutas y clases específicas del problema.
=============================================================================
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# =============================================================================
# Configuración y Constantes
# =============================================================================

# Clases del problema según la especificación
CLASSES = {
    0: {"name": "WBC", "color": "blue"},
    1: {"name": "RBC", "color": "red"},
    2: {"name": "Platelets", "color": "green"}
}

# URL de ejemplo (El estudiante o profesor debe colocar el enlace directo de descarga del zip de Roboflow)
# Nota: En Roboflow, se puede obtener un link de descarga directa exportando en formato YOLOv8.
DATASET_URL = "https://public.roboflow.com/ds/LynAPgGHGA?key=iG57RBcnRf"
DATASET_DIR = Path("./dataset")

# =============================================================================
# Módulo de Gestión de Datos (Single Responsibility Principle)
# =============================================================================

class DatasetManager:
    """Responsable exclusivamente de verificar, descargar y extraer los datos."""
    
    def __init__(self, url: str, target_dir: Path):
        self.url = url
        self.target_dir = target_dir

    def is_downloaded(self) -> bool:
        """Verifica si la estructura básica del dataset ya existe."""
        if not self.target_dir.exists():
            return False
        # Verificar particiones esperadas
        for split in ['train', 'valid', 'test']:
            if not (self.target_dir / split / 'images').exists():
                return False
        return True

    def download_and_extract(self):
        """Descarga el archivo zip y lo extrae."""
        if self.is_downloaded():
            print(f"El conjunto de datos ya se encuentra en {self.target_dir}.")
            return

        print("Iniciando descarga del conjunto de datos...")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.target_dir / "dataset.zip"

        try:
            urllib.request.urlretrieve(self.url, zip_path)
            print("Descarga completada. Extrayendo archivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.target_dir)
            print("Extracción completada.")
        except Exception as e:
            print(f"Error durante la descarga o extracción: {e}")
        finally:
            if zip_path.exists():
                os.remove(zip_path) # Limpieza del archivo comprimido

# =============================================================================
# Módulo de Lectura de Anotaciones
# =============================================================================

class AnnotationReader:
    """Responsable de leer y procesar las etiquetas en formato YOLO."""
    
    @staticmethod
    def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """Lee un archivo .txt de YOLO y retorna las cajas delimitadoras."""
        boxes = []
        if not label_path.exists():
            return boxes
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append((class_id, cx, cy, w, h))
        return boxes

# =============================================================================
# Módulo de Visualización e Interacción
# =============================================================================

class DatasetViewer:
    """Responsable de la interfaz gráfica y el manejo de eventos del teclado."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.splits = ['train', 'valid', 'test']
        self.current_split_idx = 0
        self.images: List[Path] = []
        self.current_image_idx = 0
        self.show_bboxes = True
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self._load_split()

    def _load_split(self):
        """Carga las rutas de las imágenes de la partición actual."""
        split_name = self.splits[self.current_split_idx]
        image_dir = self.base_dir / split_name / 'images'
        
        if image_dir.exists():
            # Soporta formatos comunes
            self.images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        else:
            self.images = []
            
        self.current_image_idx = 0
        self._update_display()

    def _draw_bboxes(self, img_width: int, img_height: int, label_path: Path):
        """Dibuja las cajas delimitadoras sobre la imagen."""
        boxes = AnnotationReader.read_yolo_labels(label_path)
        for class_id, cx, cy, w, h in boxes:
            if class_id not in CLASSES:
                continue
                
            # Convertir coordenadas YOLO (normalizadas) a coordenadas matplotlib (píxeles)
            # YOLO: (centro_x, centro_y, ancho, alto) -> Matplotlib: (x_min, y_min, ancho, alto)
            w_px, h_px = w * img_width, h * img_height
            x_min = (cx * img_width) - (w_px / 2)
            y_min = (cy * img_height) - (h_px / 2)
            
            color = CLASSES[class_id]["color"]
            label_name = CLASSES[class_id]["name"]
            
            rect = patches.Rectangle((x_min, y_min), w_px, h_px, 
                                     linewidth=1.5, edgecolor=color, facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(x_min, y_min - 5, label_name, color=color, 
                         fontsize=9, weight='bold', backgroundcolor='white')

    def _update_display(self):
        """Actualiza la figura de Matplotlib con la imagen y estado actuales."""
        self.ax.clear()
        split_name = self.splits[self.current_split_idx]
        
        if not self.images:
            self.ax.text(0.5, 0.5, f"No se encontraron imágenes en la partición '{split_name}'.", 
                         ha='center', va='center', fontsize=12)
            self.ax.axis('off')
            self.fig.canvas.draw()
            return

        # Cargar imagen
        img_path = self.images[self.current_image_idx]
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        self.ax.imshow(img)
        self.ax.axis('off')
        
        # Título descriptivo
        title = (f"Partición: {split_name.upper()} | Imagen {self.current_image_idx + 1}/{len(self.images)}\n"
                 f"Archivo: {img_path.name}\n"
                 f"Controles: [←/→] Navegar | [1/2/3] Cambiar Partición | [b] Alternar Cajas")
        self.ax.set_title(title, fontsize=10, pad=10)

        # Buscar archivo de etiquetas correspondiente y dibujar
        if self.show_bboxes:
            # Asume que las etiquetas están en ../labels/ con el mismo nombre y extensión .txt
            label_dir = self.base_dir / split_name / 'labels'
            label_path = label_dir / f"{img_path.stem}.txt"
            self._draw_bboxes(img_width, img_height, label_path)

        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Maneja las interacciones del teclado."""
        if event.key == 'right':
            if self.images and self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                self._update_display()
        elif event.key == 'left':
            if self.images and self.current_image_idx > 0:
                self.current_image_idx -= 1
                self._update_display()
        elif event.key == 'b':
            self.show_bboxes = not self.show_bboxes
            self._update_display()
        elif event.key == '1':
            self.current_split_idx = 0
            self._load_split()
        elif event.key == '2':
            self.current_split_idx = 1
            self._load_split()
        elif event.key == '3':
            self.current_split_idx = 2
            self._load_split()

# =============================================================================
# Punto de Entrada
# =============================================================================

def main():
    manager = DatasetManager(url=DATASET_URL, target_dir=DATASET_DIR)
    
    # 1 y 2: Verificar y descargar
    if not manager.is_downloaded():
        if DATASET_URL == "COLOQUE_AQUI_EL_ENLACE_DIRECTO_AL_ZIP_DE_ROBOFLOW":
            print("Error: Debe configurar 'DATASET_URL' con el enlace válido de Roboflow antes de ejecutar.")
            return
        manager.download_and_extract()
    else:
        print("El conjunto de datos ya está listo localmente.")

    # 3: Lanzar explorador visual
    print("\nLanzando explorador del conjunto de datos...")
    print("Asegúrese de seleccionar la ventana de la imagen para usar el teclado.")
    _ = DatasetViewer(base_dir=DATASET_DIR)
    plt.show()

if __name__ == "__main__":
    main()
