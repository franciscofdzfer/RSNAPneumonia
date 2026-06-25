# RSNA Pneumonia Detection Challenge
 
Proyecto de grupo para [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge).
 
**Integrantes:** [Anyi Lorena Pérez] · [Francisco Fernandez]
 
## Resumen del problema
 
Detección de neumonía en radiografías de tórax (formato DICOM). El objetivo es predecir bounding boxes alrededor de zonas de opacidad pulmonar compatibles con neumonía. Evaluado con mAP sobre un rango de umbrales de IoU.
 
Ver [`docs/informe_proyecto.md`](docs/informe_proyecto.md) para el detalle completo del análisis, decisiones y resultados por fase.


## Resultados (se actualiza conforme avanza el proyecto)
 
| Experimento | Métrica local | Score Kaggle | Notas |
|---|---|---|---|
| Clasificación binaria (ResNet18) | AUC 0.876 | N/A (no aplica a competencia) | Validación de pipeline |
| RetinaNet v1 | — | — | En entrenamiento |
| ResNet50 backbone congelado | AUC 0.7763 | 0.0 | Fase 2 - GradCAM bbox |
| ResNet50 backbone descongelado | AUC 0.8341 | 0.0 | Fase 2 - GradCAM bbox |
| ResNet50 fine-tuning 2 fases | AUC 0.8262 | 0.0 | Fase 2.1 - GradCAM bbox |



https://www.kaggle.com/datasets/franciscofdzfer/rsna-png-512-test

https://www.kaggle.com/datasets/franciscofdzfer/rsna-png-512
