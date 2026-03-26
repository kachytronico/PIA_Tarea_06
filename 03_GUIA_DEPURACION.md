# Guía de depuración — Errores comunes y cómo resolverlos

Este documento recoge los errores que pueden aparecer al ejecutar el notebook `PIA_06_Tarea_Alfredo_FINAL.ipynb` en Google Colab, con su causa y el fix exacto.

---

## Antes de ejecutar — Checklist

Antes de dar a "Ejecutar todo" o empezar celda a celda, verifica esto:

- [ ] GPU activada: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4`
- [ ] Ejecutar las celdas **en orden**, de arriba a abajo
- [ ] Si una celda da error, **no ejecutar las siguientes** hasta resolverlo

---

## Errores por celda

### Celda 01 — Instalación de librerías

**Error:** `ERROR: pip's dependency resolver does not currently take into account...`  
**Causa:** Conflictos de versiones entre librerías de Colab. No es un error real.  
**Fix:** Ignorar y continuar. Si la celda termina con `Successfully installed`, todo va bien.

**Error:** La celda termina sin instalar nada.  
**Causa:** Las librerías ya estaban instaladas.  
**Fix:** Normal. Continuar.

---

### Celda 07 — Carga del dataset

**Error:** `ConnectionError` o `requests.exceptions.ReadTimeout`  
**Causa:** La conexión a HuggingFace falló (red inestable o servidor ocupado).  
**Fix:** Volver a ejecutar la celda. Si persiste, esperar 1-2 minutos y reintentar.

**Error:** `FileNotFoundError` o `DatasetNotFoundError`  
**Causa:** El nombre del dataset está mal escrito.  
**Fix:** Verificar que la línea dice exactamente `load_dataset("SetFit/toxic_conversations")`.

---

### Celda 20 — Función de traducción

**Error:** `SyntaxError: invalid syntax` en una línea con texto en español  
**Causa:** Hay texto de conclusión metido dentro de la celda de código, sin `#`.  
**Fix:** Busca el texto en español al final de la celda (sin comillas ni `#`), selecciónalo todo y bórralo. Ese texto va en la celda Markdown de conclusión, no en el código.

**Error:** `OSError: Can't load tokenizer for 'Helsinki-NLP/opus-mt-es-en'`  
**Causa:** La descarga del modelo falló o la celda 18 no se ejecutó.  
**Fix:** Ejecutar primero la celda 18 y luego la 20.

---

### Celda 23 — Embedding general GloVe

**Error:** `ValueError: unknown model 'glove-twitter-25'`  
**Causa:** El nombre del modelo de gensim está mal.  
**Fix:** Los modelos disponibles son `glove-twitter-25`, `glove-twitter-50`, `glove-twitter-100`, `glove-wiki-gigaword-50`. Verificar que el código usa exactamente uno de estos.

**Error:** La descarga tarda mucho (más de 10 minutos)  
**Causa:** `glove-twitter-100` pesa ~387 MB. Normal.  
**Fix:** Usar `glove-twitter-25` (66 MB) si hay problemas de tiempo o memoria.

---

### Celda 27 — Embedding propio Word2Vec

**Error:** `MemoryError`  
**Causa:** El corpus es demasiado grande para la RAM disponible.  
**Fix:** Reducir el tamaño de `train_df` antes de entrenar. Añadir esta línea antes del Word2Vec:
```python
train_df_w2v = train_df.sample(n=50000, random_state=random_seed)
```
Y usar `train_df_w2v["text"]` en lugar de `train_df["text"]` para el corpus.

---

### Celda 31 — Random Forest

**Error:** `ValueError: Input contains NaN`  
**Causa:** Algunas frases no tienen ningún token en el vocabulario y `frase_a_vector_general()` devuelve NaN en lugar de zeros.  
**Fix:** Verificar que la función tiene el fallback de zeros:
```python
if len(vectores) == 0:
    return np.zeros(embedding_general.vector_size)
```

**Error:** La celda tarda más de 30 minutos  
**Causa:** Está vectorizando todo el test_df (puede ser muy grande tras el split original).  
**Fix:** Reducir el test para la evaluación del RF:
```python
test_df_rf = test_df.sample(n=10000, random_state=random_seed)
X_test_emb  = np.array([frase_a_vector_general(t) for t in test_df_rf["text"]])
y_test      = test_df_rf["label"].values
```

---

### Celdas 35-38 — Clasificador Deep Learning AWD-LSTM

**Error:** `CUDA out of memory`  
**Causa:** El batch_size `bs=64` es demasiado grande para la GPU con el vocabulario generado.  
**Fix:** Cambiar `bs=64` a `bs=32` en la línea:
```python
dls_deep = db_deep.dataloaders(train_df_fastai, bs=32)
```

**Error:** `AttributeError: 'DataLoaders' object has no attribute 'show_batch'`  
**Causa:** El DataBlock no se construyó correctamente.  
**Fix:** Ejecutar de nuevo la celda 35 completa antes de continuar.

**Comportamiento:** El accuracy sube muy lento o se queda estancado en 0.50  
**Causa:** El learning rate no es el adecuado.  
**Fix:** Usar el valor del valle que devuelve `lr_find()` en lugar de `1e-3`. Si el valle está en `4e-3`, usar ese valor en `fine_tune`.

---

### Celdas 41-46 — Language Model y Encoder

**Error:** `FileNotFoundError: encoder_toxicidad.pth not found`  
**Causa:** La celda 43 (`save_encoder`) no se ejecutó o falló.  
**Fix:** Ejecutar la celda 43 antes de la 44. Verificar que el output de la 43 dice `✅ Encoder guardado como 'encoder_toxicidad'`.

**Error:** `RuntimeError: Expected all tensors to be on the same device`  
**Causa:** El language model se entrenó en CPU y el clasificador intenta usar GPU (o viceversa).  
**Fix:** Asegurarse de que ambos usan `.to_fp16()` y que la GPU está activa.

---

### Celda 49 — Comparativa

**Error:** `NameError: name 'acc_clasico' is not defined`  
**Causa:** El Random Forest (celda 32) no se ejecutó o falló.  
**Fix:** Ejecutar todas las celdas de evaluación anteriores en orden.

**Error:** `NameError: name 'acc_deep' is not defined`  
**Causa:** La celda 38 de evaluación del AWD-LSTM no se ejecutó.  
**Fix:** Ejecutar celda 38.

---

### Celda 52 — Bot administrador

**Error:** `NameError: name 'learner_lm' is not defined`  
**Causa:** El clasificador con encoder (celdas 44-45) no se ejecutó.  
**Fix:** Ejecutar las celdas del clasificador LM en orden.

**Error:** `RuntimeError` al llamar a `learner_lm.predict()`  
**Causa:** El DataLoader del learner apunta al conjunto de test en lugar de al de train.  
**Fix:** Restaurar el DataLoader original:
```python
learner_lm.dls = dls_clf_lm
```
Y luego volver a llamar al bot.

---

## Errores que NO son errores

Estos mensajes aparecen habitualmente y se pueden ignorar:

```
No se pudo representar el contenido de 'application/vnd.jupyter.widget-view+json'
```
→ Son las barras de progreso de HuggingFace. En VS Code no se renderizan. En Colab sí. No afectan al resultado.

```
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub
```
→ El modelo se descarga igual sin autenticación para modelos públicos. Ignorar.

```
Repo card metadata block was not found. Setting CardData to empty.
```
→ Aviso del dataset de HuggingFace. No afecta a los datos. Ignorar.

```
The tied weights mapping and config for this model...
```
→ Aviso de MarianMT. Normal. Ignorar.

---

## Si el runtime se reinicia solo

Colab reinicia el entorno cuando se queda sin memoria. Señales: las variables desaparecen, los números de celda vuelven a `[ ]`.

**Fix:** Ejecutar todas las celdas de nuevo desde la primera, en orden. No saltar ninguna.

Si el problema se repite con frecuencia, reducir el tamaño de los datos:
- En la celda de downsampling, bajar el `threshold` de `1.2` a `0.5` para usar menos datos en entrenamiento.
- En el RandomForest, usar solo 10.000 ejemplos de test como se indicó arriba.
