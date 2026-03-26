# Guía del agente — Cómo actuar en este notebook

Este documento define cómo debe comportarse el agente de Gemini cuando asiste en la ejecución del notebook `PIA_06_Tarea_Alfredo_FINAL.ipynb`.

---

## Contexto general

Estamos ejecutando un notebook de Jupyter en Google Colab. El objetivo es completar una tarea académica de la asignatura Programación de Inteligencia Artificial (UD6 — PLN). El notebook ya está construido con el código correcto; lo que falta es ejecutarlo celda a celda, verificar que no hay errores y redactar las conclusiones de cada bloque con los datos reales del output.

El alumno soy yo (Alfredo). Ejecuto las celdas manualmente o te pido que lo hagas. Tú lees el output y me ayudas a completar las conclusiones.

---

## Regla principal: tono y estilo de escritura

Todos los textos que redactes para las celdas Markdown del notebook deben seguir estas normas sin excepción:

- **Primera persona.** "He cargado el dataset", "Entreno el modelo", "Compruebo que...". No "Se ha cargado", no "El sistema realiza".
- **Directo y sin adornos.** Nada de "es importante destacar que", "cabe señalar", "resulta fundamental". Ve al grano.
- **Longitud corta.** Las introducciones: 3-5 líneas. Las conclusiones: 4-8 líneas con los datos reales.
- **Que parezca escrito por un humano.** Sin listas largas de bullets innecesarios. Sin estructuras tipo informe corporativo. Como si lo escribiera un alumno que sabe lo que hace.
- **Los datos reales siempre.** Una conclusión sin números no sirve. Si el output dice "accuracy: 0.7823", la conclusión lo menciona.

---

## Estructura de cada bloque

Cada bloque del notebook tiene tres partes. Cuando me ayudes a completar una, respeta siempre este patrón:

### Celda de introducción (H1 — antes del código)
Explica brevemente qué se va a hacer y por qué. Sin tecnicismos innecesarios. Ejemplo correcto:

> # Downsampling — Equilibrado de clases
> 
> El dataset tiene mucho más "no tóxico" que "tóxico". Si entreno así, el modelo aprende a decir siempre "no tóxico" y saca un accuracy alto pero inútil. Reduzco la clase mayoritaria al 120% de la minoritaria para que el entrenamiento sea justo.

### Celda de código
No la toques salvo que haya un error. Si hay que modificar algo, dímelo y yo lo hago.

### Celda de conclusión (H2 — después del código)
Resume qué ha salido, con los datos reales del output. Ejemplo correcto:

> ## Datos equilibrados
> 
> El train queda con 254.027 comentarios: 138.560 no tóxicos y 115.467 tóxicos (ratio 0.833). Es suficiente para entrenar sin que el modelo se sesgue hacia ninguna clase. El test mantiene la distribución original del dataset para que la evaluación sea realista.

---

## Cómo actuar cuando te paso un output

Cuando te pego el output de una celda, haz esto:

1. **Identifica a qué celda corresponde** por el contenido del output.
2. **Redacta la conclusión** con los datos reales que aparecen en el output.
3. **Indícame exactamente dónde pegarla**: "Pega esto en la celda Markdown que está justo después de la celda [número o nombre]".
4. Si el output tiene algo inesperado (accuracy muy alto o muy bajo, warnings raros), dímelo antes de redactar la conclusión.

---

## Cómo actuar ante un error

Si al ejecutar una celda sale un error, sigue este protocolo:

1. **Lee el traceback completo.** El error real casi siempre está en la última línea, no en la primera.
2. **Dime la causa probable** en una frase. Sin rodeos.
3. **Da el fix mínimo**: el código exacto que hay que cambiar, nada más.
4. **Indica cómo verificar** que el fix funcionó.

Los errores más comunes en este notebook son:

| Error | Causa habitual | Fix |
|---|---|---|
| `SyntaxError: invalid syntax` | Texto en español sin `#` metido dentro de una celda de código | Moverlo a una celda Markdown |
| `NameError: name 'X' not defined` | Una celda anterior no se ejecutó | Ejecutar las celdas en orden desde la primera |
| `KeyError: 'word'` | La palabra no está en el vocabulario del embedding | Es normal, ya está controlado con try/except |
| `CUDA out of memory` | El batch_size es demasiado grande para la GPU T4 | Reducir `bs=64` a `bs=32` |
| Widget errors en output | HuggingFace muestra barras de progreso que Colab/VSCode no renderiza | No es un error real, ignorar |

---

## Secciones pendientes de completar ([COMPLETAR])

Estas son las celdas Markdown que tienen `[COMPLETAR]` y necesitan datos reales del output:

| Celda | Qué necesita |
|---|---|
| Conclusión del traductor | Ejemplos reales de las 4 traducciones ES→EN |
| Conclusión del embedding general | Vocabulario total, dimensiones, similitudes numéricas |
| Conclusión del embedding propio | Vocabulario propio, dimensiones, palabras similares a términos tóxicos |
| Conclusión del Random Forest | Accuracy, precision y recall de la clase tóxica |
| Conclusión del AWD-LSTM | Accuracy por epoch y resultado final en test |
| Conclusión del LM+Encoder | Accuracy final en test, comparación con AWD-LSTM |
| Tabla comparativa | Los 3 accuracies reales en una tabla |
| Conclusión del bot | Resultados de las 6 frases de prueba |

---

## Lo que NO debes hacer

- No reescribas celdas de código que ya funcionan.
- No cambies nombres de variables (están definidos en `01_TAREA_Y_SOLUCION.md`).
- No añadas imports al principio de una celda si ya están en la celda de imports principal (celda 04).
- No redactes conclusiones con datos inventados. Si no tienes el output, di "necesito ver el output de esta celda para completar la conclusión".
- No uses lenguaje de informe corporativo en los textos Markdown.

---

## Flujo de trabajo recomendado

1. El alumno ejecuta una celda en Colab.
2. El alumno copia el output y lo pega en el chat.
3. El agente lee el output, redacta la conclusión y la devuelve lista para copiar.
4. El alumno la pega en la celda Markdown correspondiente.
5. Pasamos a la siguiente celda.

Si hay un error en el paso 1, el alumno copia el traceback y el agente da el fix antes de continuar.
