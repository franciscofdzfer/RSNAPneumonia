# Guión de Presentación — RSNA Pneumonia Detection

---

## Slide 1 — Portada

Este proyecto es una participación tardía en la competencia de Kaggle RSNA Pneumonia Detection.
El objetivo era detectar neumonía en radiografías de tórax reales anotadas por radiólogos.
Trabajamos en paralelo con dos enfoques distintos para luego comparar resultados.
El entorno fue Kaggle Notebooks con GPU T4, sin coste adicional para el equipo.

---

## Slide 2 — El Problema y EDA

El problema no es clasificación sino detección de objetos: hay que localizar la zona afectada.
La métrica mAP promediada sobre IoU 0.40–0.75 penaliza cajas imprecisas aunque la zona sea correcta.
El dataset tiene un fuerte desbalance: solo el 22.5% de las imágenes tienen neumonía.
Esto obligó a usar técnicas como pos_weight y Focal Loss para compensar durante el entrenamiento.

---

## Slide 3 — EDA Visual

Las radiografías con Target=1 muestran las cajas rojas anotadas por radiólogos certificados.
El tamaño de las cajas tiene un pico alrededor de 200px de ancho sobre imágenes de 1024px.
La vista AP presenta más positivos que la PA, pero es un sesgo de selección clínico.
Los pacientes en AP suelen estar más graves, no es que la vista cause más neumonía.

---

## Slide 4 — Estrategia del Equipo

Trabajamos en paralelo para poder comparar dos familias de modelos diferentes al final.
Lorena se centró en RetinaNet, un detector nativo con Focal Loss para el desbalance.
Fran exploró primero GradCAM como localizador indirecto y luego YOLO como detector real.
El criterio de selección final fue el score de Kaggle, no la métrica interna.

---

## Slide 5 — Enfoque A: RetinaNet

RetinaNet es uno de los detectores más usados en las soluciones ganadoras de esta competencia.
El backbone se adaptó de 3 canales RGB a 1 canal para las radiografías en escala de grises.
Con solo 5 épocas el modelo aprendió a discriminar positivos de negativos correctamente.
El score promedio top-1 fue 0.148 en imágenes con neumonía frente a 0.092 en negativas.
El tiempo total de entrenamiento fue aproximadamente 3.3 horas en una GPU T4.

---

## Slide 6 — Calibración de Umbral

Con el umbral por defecto de 0.3 la submission salió con el 97% de filas vacías.
El modelo sí aprendía, pero con pocas épocas las confianzas absolutas son bajas.
Se probaron 6 valores de umbral comparando el porcentaje de predicciones con el 22.5% esperado.
El mAP local calculado sobre validación interna confirmó que 0.05 era el mejor umbral.
El resultado fue mixto: mejoró en public pero empeoró en private por alta varianza.

---

## Slide 7 — Enfoque B: YOLOv11m

YOLO aprende directamente a predecir bounding boxes, a diferencia de GradCAM que es indirecto.
Se entrenaron 30 épocas sobre el dataset completo de 26,684 imágenes en formato PNG.
Las imágenes de 1 canal se convirtieron a 3 canales replicando la misma información.
El mAP50 de validación fue 0.340, que es razonable para un primer entrenamiento completo.
El threshold se fijó en 0.05 tras comprobar que las confianzas máximas eran de ~0.19.

---

## Slide 8 — Comparativa de Resultados

YOLO supera a RetinaNet en ambos splits del leaderboard de Kaggle.
La diferencia más notable es en el score privado: +36.9% a favor de YOLO.
RetinaNet con threshold 0.05 mejoró en public pero empeoró en private respecto a 0.10.
Esta inestabilidad es consecuencia directa de entrenar pocas épocas con validación limitada.
El promedio de ambos splits favorece claramente a YOLO: 0.09925 frente a 0.08260.

---

## Slide 9 — Lecciones Aprendidas

GradCAM es una técnica de interpretabilidad, no un localizador preciso de objetos.
Nunca supera el umbral de IoU mínimo exigido por la competencia, que es 0.40.
Mixed precision no funciona con todas las arquitecturas: Focal Loss es inestable en FP16.
Validar el pipeline con un baseline simple antes de la arquitectura final ahorra tiempo.
El mAP local es mejor que cualquier heurística, pero no garantiza generalizar al leaderboard.

---

## Slide 10 — Conclusiones

YOLO es el modelo final del equipo con public 0.09126 y private 0.10725.
Ambos modelos aprendieron la tarea correctamente pero necesitan más épocas de entrenamiento.
La varianza entre splits public y private es alta cuando el modelo tiene poca confianza calibrada.
Los próximos pasos son más épocas, k-fold para estimación robusta y ensemble de los dos modelos.
