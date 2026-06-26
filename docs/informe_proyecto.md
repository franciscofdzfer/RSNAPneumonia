# Informe de Proyecto — RSNA Pneumonia Detection Challenge

**Integrantes:** [Lorena Pèrez Rodas] / [Francisco Fernandez]
**Asignatura:** [Deep Learning]
**Repositorio:** [link a GitHub] · **Tablero:** (https://trello.com/invite/b/6a3a7b7e28bc818e1d4cdc52/ATTI0a78624097e6d7239ea5a40aed75fd74E1C9648F/rsna-pneumonia)
**Competencia:** [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

## Bloque 1 — Entendimiento del problema y EDA
 
**El problema:** detección de objetos sobre radiografías de tórax en formato DICOM (1024×1024, escala de grises). Para cada imagen hay que predecir 0 o más bounding boxes de opacidad pulmonar compatible con neumonía, cada una con un score de confianza. Formato de envío: `patientId, "confidence x y width height ..."`.
 
**Métrica:** mean Average Precision (mAP) promediado sobre un rango de umbrales de IoU (0.40–0.75). Una predicción solo cuenta como acierto si su solapamiento con una caja real supera el umbral evaluado — cajas aproximadas no bastan, hay que ajustar tamaño y posición.
 
**Datos:** ~26,684 imágenes de entrenamiento. `stage_2_train_labels.csv` tiene una fila *por caja* (no por paciente); `stage_2_detailed_class_info.csv` da la clase detallada (`Normal` / `No Lung Opacity-Not Normal` / `Lung Opacity`).
 
**Hallazgos clave del EDA:**
 
| Aspecto | Hallazgo |
|---|---|
| Balance de clases | 77.5% sin neumonía / 22.5% con neumonía → dataset desbalanceado |
| Cajas por paciente positivo | Mayoría con 1–2 cajas; tamaño promedio ≈ 218×329 px (sobre 1024×1024) |
| Vista (AP/PA) | AP muestra más neumonía que PA — sesgo de selección, no causalidad clínica (AP suele tomarse a pacientes encamados/más graves) |
| Edad | Tasa de positivos estable 20–70 años (~20–25%), algo mayor en extremos de edad |
 
**Decisiones derivadas:** resize a 512×512; usar `pos_weight`/Focal Loss para compensar el desbalance.
 
**Arquitecturas investigadas:** RetinaNet (Focal Loss, maneja bien el desbalance) y Mask R-CNN dominan las soluciones top de esta competencia, con backbones ResNet-50/101 adaptados de 3 a 1 canal. Hallazgo de equipos ganadores: encoger ~10-17% las cajas predichas mejora el score (las cajas de test tienden a ser más pequeñas que las de train).
 
---
 
## Bloque 2 — Modelado: enfoques individuales
 
Se trabajó en paralelo para luego comparar y quedarnos con el mejor enfoque (sin ensemble).
 
### 2.1 Enfoque A — Lorena: RetinaNet (1 canal) + baseline de clasificación
 
**Paso previo — clasificación binaria (validación de pipeline):**
ResNet18 (1 canal, pesos RGB promediados), `BCEWithLogitsLoss` con `pos_weight`, split 85/15 estratificado.
 
| Métrica | Valor |
|---|---|
| AUC (validación) | **0.876** |
| Sensibilidad / Especificidad (umbral 0.5) | 75.8% / 81.8% |
| Precision (umbral 0.5) | 54.8% |
| Umbral óptimo (Youden's J) | 0.389 |
 
Confirma que el pipeline (lectura DICOM → modelo → evaluación) funciona correctamente antes de abordar detección real.
 
**Detección — RetinaNet (`retinanet_resnet50_fpn_v2`, backbone 1 canal):**
 
| Parámetro | Valor |
|---|---|
| Tamaño de imagen | 512×512 |
| Épocas | 5 (completas) |
| Batch size | 6 |
| Optimizer | SGD (lr=0.005, momentum=0.9) + warm-up lineal (200 iter) |
| Precisión | FP32 |
| Tiempo total de entrenamiento | ~3.3h (1 GPU T4) |
 
**Resultados de detección:**
- `train_loss` bajó de forma estable y sin explosión: **0.2136** al final de la época 5 (sin NaN ni inestabilidad).
- Diagnóstico de scores en validación: el modelo discrimina correctamente entre imágenes con y sin neumonía — score promedio top-1 de **0.148** en imágenes con neumonía real vs. **0.092** en negativas (muestra de validación, n=17 imágenes). El modelo aprendió la tarea, pero con solo 5 épocas no alcanzó confianza calibrada alta (scores bajos en términos absolutos).
- **Umbral de confianza (`SCORE_THRESHOLD`) ajustado mediante búsqueda sistemática**, no a ojo: se probaron 6 valores (0.08–0.20) comparando el % de filas con predicción contra el ~22.5% de positivos esperado según el EDA. Resultado:
| Umbral | Filas con predicción | % | Diferencia vs. esperado (22.5%) |
|---|---|---|---|
| 0.08 | 1080/3000 | 36.0% | +13.5 pp |
| **0.10** | **741/3000** | **24.7%** | **+2.2 pp (mejor)** |
| 0.12 | 525/3000 | 17.5% | −5.0 pp |
| 0.15 | 339/3000 | 11.3% | −11.2 pp |
| 0.18 | 254/3000 | 8.5% | −14.0 pp |
| 0.20 | 212/3000 | 7.1% | −15.4 pp |
 
Se eligió **`SCORE_THRESHOLD = 0.10`** por ser el más cercano a la proporción esperada de positivos.
 
**Submission final:** se realizaron 2 envíos a Kaggle para comparar umbrales (ver detalle de mAP local más abajo). Resultados:
 
| Umbral | Filas con predicción | Public | Private |
|---|---|---|---|
| 0.10 | 741/3000 (24.7%) | 0.05555 | 0.10379 |
| 0.05 | 1664/3000 (55.5%) | 0.08686 | 0.07834 |
 
**Refinamiento del umbral mediante mAP local:** la elección inicial de `SCORE_THRESHOLD=0.10` se basó en una heurística (acercar el % de filas con predicción al 22.5% de positivos esperado según el EDA) — útil para una primera estimación, pero no equivalente a maximizar mAP directamente, ya que esta métrica depende de la calidad del IoU por imagen, no de una proporción global. Se implementó una réplica local del cálculo oficial de mAP (promediado sobre IoU 0.40–0.75, ver `calcular_map_local.py`), validada con casos sintéticos de resultado conocido (IoU y AP exactos), y se comparó contra el propio set de validación:
 
| SCORE_THRESHOLD | mAP local (validación interna) |
|---|---|
| **0.05** | **0.2378 (mejor)** |
| 0.08 | 0.2187 |
| 0.10 | 0.1989 |
| 0.12 | 0.1737 |
| 0.15 | 0.1435 |
 
La tendencia es monótona: a menor umbral, mayor mAP local — coherente con el diagnóstico previo de que el modelo, tras solo 5 épocas, todavía no tiene confianza calibrada alta y se beneficia de priorizar recall sobre precisión en esta etapa. Nota: el valor absoluto del mAP local no es comparable directamente al score del leaderboard de Kaggle (se calcula sobre el split interno de validación, no sobre el test set oculto de la competencia) — solo es válido para comparar la *tendencia relativa* entre umbrales.
 
**Resultado del segundo submission (`SCORE_THRESHOLD=0.05`):**
 
| Umbral | Public | Private | Promedio |
|---|---|---|---|
| 0.10 | 0.05555 | **0.10379** | 0.07967 |
| 0.05 | **0.08686** | 0.07834 | 0.08260 |
 
El resultado fue **mixto, no una mejora limpia**: el public score mejoró notablemente (+56.4%), pero el private score empeoró (−24.5%) respecto al umbral 0.10. El promedio de ambos splits favorece ligeramente a 0.05 (+3.7%), muy por debajo del +19.6% que sugería el mAP local.
 
**Interpretación:** el mAP local predijo correctamente la dirección de la mejora en el split público, pero no en el privado. Esto es una limitación real, no un error de cálculo: con un modelo de solo 5 épocas y un split de validación interno relativamente pequeño, la varianza entre subconjuntos del test set (public/private) es alta, y un único split de validación no garantiza generalización idéntica a ambos. La metodología (mAP local en vez de heurísticas) sigue siendo la decisión correcta — calcular la métrica real es mejor que adivinar — pero el caso ilustra que con poco entrenamiento el modelo es sensible a pequeños cambios de umbral de forma no totalmente predecible. Se mantiene el submission con `SCORE_THRESHOLD=0.05` como entrega final por tener mejor promedio agregado y mejor desempeño en el split público.
 
### 2.2 Enfoque B — Fran: ResNet50+GradCAM (exploratorio) y YOLO (entregado)
 
**Variante 1 — ResNet50 + GradCAM (clasificación + localización indirecta):**
En vez de regresión directa de bounding boxes (como RetinaNet), este enfoque entrena un clasificador binario (ResNet50) y usa **GradCAM** para generar un mapa de activación que indica qué regiones de la imagen influyeron más en la predicción de "neumonía" — las cajas se infieren a partir de ese mapa de calor, no se predicen directamente. Tratamiento de imagen: conversión de la radiografía (1 canal) a 3 canales replicando la misma información en cada canal (RGB "falso", sin preprocesamiento adicional tipo CLAHE).
 
Esta variante quedó como **prueba exploratoria, no se completó/entregó** — sin métrica ni submission asociados. [completar si se retoma: motivo de no completarla, resultados parciales si los hay]
 
**Variante 2 — YOLO (entregada y comparada):**
 
| Parámetro | Valor |
|---|---|
| Arquitectura | YOLO v11m |
| Tratamiento de imagen | 1 canal replicado a 3 canales idénticos (RGB fake), formato PNG |
| Épocas | 30 |
| Tamaño de imagen | 512 x 512 |
 
**Resultado (YOLO):**
 
| Métrica | Valor |
|---|---|
| Public score | 0.09126 |
| Private score | **0.10725** |
 
---
 
## Bloque 3 — Comparación y selección del mejor enfoque
 
| Enfoque | Arquitectura | Public | Private | Promedio |
|---|---|---|---|---|
| A | RetinaNet (1 canal, 5 épocas) | 0.08686 | 0.07834 | 0.08260 |
| B.1 | ResNet50 + GradCAM | No completado (exploratorio) | — | — |
| **B.2** | **YOLO** | **0.09126** | **0.10725** | **0.09925 (mejor)** |
 
**Criterio de selección:** se prioriza el enfoque con mejor score de leaderboard real (promedio public/private), dado que no se implementó mAP local para el Enfoque B y por tanto la comparación más justa entre ambos integrantes es sobre el resultado de Kaggle. Se descarta ensemble por restricciones de tiempo del proyecto.
 
**Decisión final:** se selecciona **YOLO (Enfoque B.2)** como modelo final del equipo. Supera al mejor resultado de RetinaNet en ambos splits del leaderboard (public: +5.1%, private: +36.9%, promedio: +20.2%). Nota sobre consistencia entre splits: YOLO tiene private > public (igual que RetinaNet con threshold=0.10), mientras que la entrega final de RetinaNet (threshold=0.05) tuvo public > private — esta inversión según el umbral elegido es justamente la inestabilidad documentada en el Bloque 4 (Enfoque A, incidencia 7), y refuerza la decisión de quedarse con YOLO como modelo final por mostrar un resultado más sólido en el split privado, que es el que determina la posición final en competencias de Kaggle. RetinaNet (Enfoque A) se documenta en detalle en este informe por el valor del proceso de debugging y la metodología de calibración de umbral (Bloques 2.1 y 4), aplicable como referencia técnica aunque no sea el modelo elegido para la entrega final.
 
---
 
## Bloque 4 — Submission y conclusiones
 
**Submission generada:** formato verificado contra `stage_2_sample_submission.csv` (3000 filas, columnas `patientId`/`PredictionString`). Se realizaron 2 envíos (`threshold=0.10` y `threshold=0.05`, ver tabla comparativa en el Bloque 2.1) — entrega final con **`SCORE_THRESHOLD=0.05`: 0.08686 (public) / 0.07834 (private)**.
 
**Incidencias técnicas relevantes (Enfoque A — proceso de debugging):**
Documentadas porque reflejan decisiones de ingeniería real, no solo el resultado final.
 
1. **Bugs de pipeline** (`UnboundLocalError` en augmentation, `NameError` por import faltante, `RuntimeError` de canales por transform interno de RetinaNet mal configurado para escala de grises) — resueltos ajustando el código del `Dataset` y los parámetros `image_mean`/`image_std`/`min_size`/`max_size` del modelo.
2. **Focal Loss inestable en FP16:** se intentó acelerar con mixed precision (AMP), pero la cabeza de clasificación de RetinaNet produjo `NaN` sistemáticamente — problema documentado en la comunidad de PyTorch (la implementación CUDA del sigmoid focal loss no es estable en precisión reducida). Se optó por FP32 puro.
3. **Explosión de la loss:** con lr=0.005 desde la primera iteración, la loss escaló de ~0.4 a >600 en pocos cientos de batches. Se resolvió con **warm-up lineal** (lr 0.0001→0.005 en 200 iteraciones) + gradient clipping (max_norm=5.0) — práctica estándar en RetinaNet/detectron2 para estabilizar el inicio del entrenamiento.
4. **Deadlock en setup multi-GPU:** se intentó paralelizar el entrenamiento entre las 2 GPUs T4 disponibles con `threading` manual (split de batch + promedio de gradientes), pero el entrenamiento se colgó sin avanzar tras ~450 batches — problema conocido de sincronización CUDA entre threads de distintos dispositivos. Se simplificó a 1 sola GPU, priorizando terminar de forma confiable sobre la velocidad.
5. **Restricciones de tiempo real:** dado el plazo de entrega, se ajustó el número de épocas (5) y se implementó checkpointing con reanudación automática para poder usar "Save & Run All" de Kaggle sin supervisión activa.
6. **Umbral de confianza mal calibrado tras pocas épocas:** con el umbral inicial (0.3, valor típico para modelos bien entrenados) el submission salió con 97% de filas vacías. Diagnóstico: el modelo sí discriminaba positivos de negativos, pero con scores bajos en términos absolutos (normal con solo 5 épocas — la confianza del modelo aún no está calibrada). Se corrigió con una búsqueda sistemática de umbral sobre el test set, comparando contra la proporción de positivos esperada según el EDA, en lugar de ajustar el valor a ojo.
7. **La heurística de umbral basada en proporción de positivos era subóptima, pero el mAP local tampoco predijo perfectamente el leaderboard real:** acercar el % de filas con predicción al 22.5% esperado (que llevó a elegir 0.10) no es equivalente a maximizar mAP. El mAP local (validado con casos sintéticos de IoU/AP conocido) sugería que `threshold=0.05` mejoraría el mAP ~19.6% — al confirmarlo con un segundo submission real, el resultado fue mixto: mejora notable en public (+56.4%) pero retroceso en private (−24.5%), con el promedio de ambos splits favoreciendo apenas a 0.05 (+3.7%). Lección: con un modelo de pocas épocas y un split de validación interno limitado, la varianza entre subconjuntos del test set es alta — el mAP local sigue siendo mejor que adivinar a ojo, pero no garantiza generalización idéntica a ambos splits del leaderboard.
**Limitaciones:**
- Solo 5 épocas de entrenamiento de RetinaNet por restricción de tiempo del proyecto — el modelo aprendió la tarea (loss estable, discrimina positivos/negativos) pero no alcanzó confianza calibrada alta; más épocas probablemente mejorarían tanto la calibración como el mAP.
- El mAP local implementado se calcula sobre un único split de validación interno (15% de train); como se observó al comparar los 2 submits reales, esto no garantiza predecir igual de bien el comportamiento en ambos splits del leaderboard (public/private) — sería deseable validación cruzada (k-fold) para una estimación más robusta.
- Sin ensemble entre los dos enfoques del equipo (se decidió priorizar el mejor individual).
- Setup multi-GPU descartado por inestabilidad (deadlock) — el entrenamiento final corrió en 1 sola GPU, más lento de lo que hubiera sido posible.
**Lecciones aprendidas:**
- Mixed precision no es "gratis": hay arquitecturas (Focal Loss) con inestabilidad numérica conocida en FP16.
- El paralelismo multi-GPU manual tiene riesgos de concurrencia no triviales; para detección de objetos, frameworks como DDP están mejor probados que soluciones ad-hoc.
- Validar el pipeline con un baseline simple (clasificación) antes de la arquitectura final ahorró tiempo de debugging.
- Una métrica local bien calculada es mejor que una heurística, pero no es garantía de generalización idéntica al leaderboard real — con poco entrenamiento, la varianza entre splits (public/private) puede ser alta.
---
 
## Reparto de responsabilidades
 
| Responsabilidad | Encargado/a |
|---|---|
| EDA y métrica de evaluación | Lorena, Fran |
| Enfoque A: RetinaNet + clasificación baseline | Lorena |
| Enfoque B: ResNet50+GradCAM (exploratorioLorena) y YOLO (modelo final elegido) | Fran |
| Comparación y selección final | Lorena, Fran |
| Informe y presentación | Lorena, Fran |
 
---
 
## Bitácora breve
 
| Fecha | Quién | Qué se hizo |
|---|---|---|
| 24/06/2026 | Lorena, Fran | EDA: distribución de clases, metadatos, bounding boxes |
| 25/06/2026 | Fran | Baseline clasificación — AUC 0.876 |
| 25/06/2026 | Lorena | RetinaNet — debugging (4 incidencias de entrenamiento) y entrenamiento final (5 épocas, train_loss 0.2136) |
| 25/06/2026 | Lorena | Diagnóstico y calibración de umbral de confianza (búsqueda sistemática) + primer submission a Kaggle |
| 25/06/2026 | Lorena | Implementación de mAP local (validado con tests sintéticos), comparación de umbrales, segundo submission de confirmación |
| 25/06/2026 | Fran | ResNet50+GradCAM (exploratorio, no completado) y YOLO — score final: 0.09126 (public) / 0.10725 (private) |
| 26/06/2026 | Lorena, Fran | Comparación de 3 resultados (RetinaNet vs. YOLO) — selección de YOLO como modelo final |
| | | |
