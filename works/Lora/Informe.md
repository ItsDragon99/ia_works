# Informe del Proyecto: Tutor IA con LoRA

## Índice
1. Objetivo General
2. Metodología y configuración
3. Resultados Esperados
4. Relación con los archivos del proyecto
5. Informe de actividades
6. Flujo resumido
7. Próximos pasos y métricas sugeridas

## Objetivo General
Desarrollar un tutor especializado en algoritmos y estructuras de datos, entrenado con datos curados, capaz de asistir a estudiantes mediante respuestas claras, contextualizadas y fáciles de ampliar con nuevos temas.

## Metodología y configuración
- Modelo base: Qwen/Qwen2.5-3B-Instruct con ajuste fino SFT + LoRA.
- Tokenización: padding a la derecha y uso del token EOS como pad.
- Entrenamiento: gradient checkpointing para eficiencia; LR 1e-4; 6 épocas; acumulación de gradientes.
- Evaluación: pruebas rápidas de generación con el adaptador LoRA antes de fusionar.
- Despliegue: servicio interactivo con plantilla de sistema en `Modelfile`.

# Resultados Esperados

1. Un tutor especializado capaz de explicar algoritmos con respuestas claras y contextualizadas.
2. Un modelo entrenado con datos cuidadosamente curados para maximizar pertinencia y precisión.
3. Una herramienta útil para estudiantes de computación que apoye el aprendizaje autónomo.
4. Un sistema adaptable para incluir nuevos temas y ampliar el temario de estudio.

## Relación con los archivos del proyecto
- `Tutor.py`: Ejecuta la inferencia interactiva utilizando el modelo base más la LoRA entrenada.
- `FineTunning.py`: Realiza el ajuste fino supervisado (SFT) con configuración de LoRA sobre el modelo base.
- `TestFineTunning.py`: Prueba rápida del adaptador LoRA para validar generación tras el entrenamiento.
- `Modelfile`: Define la plantilla del sistema para servir el modelo ya compilado/convertido.
- `merge_lora.py`: Fusiona el modelo base con los pesos LoRA y guarda el resultado para despliegue.

## Informe de actividades
- Preparación de datos (`tutor_v2.jsonl`) y definición de formato de prompt/respuesta.
- Ajuste fino con LoRA usando `FineTunning.py` sobre Qwen2.5-3B, habilitando gradient checkpointing para eficiencia.
- Validación rápida del adaptador con `TestFineTunning.py`, generando respuestas de control.
- Fusión opcional de pesos con `merge_lora.py` para obtener un modelo consolidado.
- Despliegue e inferencia interactiva vía `Tutor.py`, utilizando el adaptador LoRA entrenado.
- Configuración de servido mediante `Modelfile`, estableciendo el rol de tutor experto en algoritmos y EDA.

## Flujo resumido
1. Entrenar (`FineTunning.py`) → guardar adaptador LoRA.
2. Probar (`TestFineTunning.py`) → verificar calidad de respuestas.
3. (Opcional) Fusionar (`merge_lora.py`) → modelo listo para despliegue.
4. Servir (`Tutor.py` + `Modelfile`) → tutor interactivo para estudiantes de computación.

## Próximos pasos y métricas sugeridas
- Ampliar el dataset con ejemplos de depuración y análisis de complejidad.
- Medir BLEU/ROUGE y coherencia humana en respuestas de prueba.
- Evaluar tiempos de inferencia en GPU/CPU y VRAM utilizada.
- Incorporar guardrails (máxima longitud de respuesta, detección de alucinaciones).
- Planificar versión GGUF actualizada tras cada fusión LoRA para despliegue ligero.
