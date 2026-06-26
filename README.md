# RSNA Pneumonia Detection Challenge
 
Proyecto de grupo para la competencia de Kaggle [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge).
 
**Integrantes:** Lorena Pérez · Francisco Fernandez
 
## Resumen del problema
 
Detección de neumonía en radiografías de tórax (formato DICOM). El objetivo es predecir bounding boxes alrededor de zonas de opacidad pulmonar compatibles con neumonía. Evaluado con mAP sobre un rango de umbrales de IoU.
 
Ver [`docs/informe_proyecto.md`](docs/informe_proyecto.md) para el detalle completo del análisis, decisiones y resultados por fase.
 
## Estructura del repositorio
 
```
.
├── notebooks/
│   ├── 01_eda.ipynb                  # Análisis exploratorio
│   ├── 02_baseline_clasificacion.ipynb   # Fase A: clasificación binaria
│   └── 03_clasificacionResNet18_deteccionRetinaNet.ipynb       # Fase B: RetinaNet
│   ├── 01_clasificacion_deteccion_yolov11.ipynb               
├── docs/
│   └── informe_proyecto.md           # Informe vivo del proyecto (Markdown)
├── outputs/
│   └── submissions/                  # CSVs de envío generados
└── README.md
```
 
> Nota: los notebooks se desarrollan y corren en Kaggle (por acceso a GPU y al dataset montado). Este repo es para versionar el código y dar seguimiento al trabajo, no para ejecutar el pipeline completo localmente.
 
## Cómo trabajamos
 
- **Tablero:** seguimiento de tareas en Trello
- **Dailies:** [horario acordado], formato qué hice / qué hago / bloqueos.
- **Flujo de Git:**
  - `main` siempre debe quedar en estado funcional.
  - Una rama por tarea/feature: `eda`, `baseline-clasificacion`, `baseline-deteccion`, etc.
  - Pull Request + revisión cruzada antes de mergear a `main`.
  - Los notebooks se exportan desde Kaggle y se suben aquí ya limpios (sin outputs gigantes si es posible, para no inflar el repo).
## Estado actual
 
| Fase | Estado |
|---|---|
| Estudio del problema (EDA, métrica, arquitecturas) | ✅ Completo |
| Separación de responsabilidades | ✅ Completo |
| Enfoque A (RetinaNet) | ✅ Completo — AUC clasificación 0.876, 2 submissions enviados |
| Enfoque B (ResNet50, YOLOv11m) | ✅ Completo — YOLOv11m entregado y comparado |
| Comparación y selección de modelo final | ✅ **YOLO seleccionado** (mejor score en ambos splits) |
| Informe final | (ver `docs/informe_proyecto.md`) |
| Presentación |  |
 
## Resultados (se actualiza conforme avanza el proyecto)
 
**Baselines de clasificación (validación de pipeline):**
 
| Modelo | AUC |
|---|---|
| ResNet18 (1 canal) — Enfoque A | **0.876** |
| ResNet50 (3 canales, descongelado) — Enfoque B | 0.834 |
| ResNet50 (3 canales, fine-tuning 2 fases) — Enfoque B | 0.778 → 0.826 |
 
**Modelos de detección (resultado real en Kaggle):**
 
| Modelo | mAP local/validación | Public | Private |
|---|---|---|---|
| RetinaNet (threshold=0.10) — Enfoque A | mAP local 0.1989 | 0.05555 | 0.10379 |
| RetinaNet (threshold=0.05) — Enfoque A | mAP local 0.2378 | 0.08686 | 0.07834 |
| **YOLOv11m — Enfoque B** | mAP50-95: 0.142 | **0.09126** | **0.10725** |
 
🏆 **Modelo final seleccionado: YOLOv11m** (mejor score en ambos splits del leaderboard).
 
## Setup rápido (si se ejecuta fuera de Kaggle)
 
```bash
pip install torch torchvision pydicom opencv-python scikit-learn pandas matplotlib seaborn
```
 
Requiere descargar el dataset desde Kaggle (`kaggle competitions download -c rsna-pneumonia-detection-challenge`) y ajustar `INPUT_DIR` en los notebooks.
 
