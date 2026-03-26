# PIA UD6 — Administrador de Comentarios para Redes Sociales
**Módulo:** Programación de Inteligencia Artificial  
**Unidad:** 6 — Procesamiento del Lenguaje Natural  
**Alumno:** Alfredo FEDE  
**Entorno de ejecución:** Google Colab con GPU T4

---

## 1. Enunciado de la tarea

El centro quiere un bot que administre los comentarios en sus redes sociales (Twitter/Instagram) y elimine automáticamente los que sean tóxicos antes de que los lean los usuarios.

No tenemos datos propios etiquetados, así que usamos el dataset público `SetFit/toxic_conversations` de HuggingFace. El problema es que ese dataset está en inglés y los comentarios de nuestras redes llegarán en español. Por eso el sistema necesita dos partes: una capa de traducción y un clasificador.

### Rúbrica oficial (10 puntos)

| Elemento | Puntos |
|---|---|
| Traductor español → inglés (no tiene por qué usar IA) | 2 pts |
| Embedding general (modelo preentrenado) | 1 pt |
| Embedding propio (entrenado con el dataset) | 1 pt |
| Clasificador con modelos clásicos | 2 pts |
| Clasificador con deep learning | 2 pts |
| Clasificador con modelo del lenguaje | 2 pts |
| **TOTAL** | **10 pts** |

---

## 2. Dataset

- **Nombre:** `SetFit/toxic_conversations`
- **Fuente:** HuggingFace Datasets
- **Carga:** `load_dataset("SetFit/toxic_conversations")`
- **Tamaño:** ~1.754.874 ejemplos de train + 50.000 de test
- **Columnas:** `text` (comentario), `label` (0=no tóxico, 1=tóxico), `label_text`
- **Idioma:** inglés
- **Desbalanceo:** ~92% no tóxico / ~8% tóxico → hay que aplicar downsampling antes de entrenar

---

## 3. Stack tecnológico

```
fastai          → clasificadores de texto (AWD-LSTM, language model)
datasets        → carga del dataset desde HuggingFace
gensim          → embeddings Word2Vec y GloVe
transformers    → modelo de traducción MarianMT
sentencepiece   → tokenizador para MarianMT
sklearn         → RandomForestClassifier, train_test_split
seaborn/matplotlib → visualizaciones
```

---

## 4. Estructura del notebook (54 celdas)

El notebook `PIA_06_Tarea_Alfredo_FINAL.ipynb` sigue esta estructura. Cada bloque tiene siempre tres partes: celda Markdown de introducción (H1) + celda(s) de código + celda Markdown de conclusión (H2).

| Índice | Tipo | Contenido |
|---|---|---|
| 00 | MD | Introducción: Instalación de librerías |
| 01 | PY | `!pip install fastai datasets gensim transformers sentencepiece` |
| 02 | MD | Conclusión: Instalación completada |
| 03 | MD | Introducción: Imports principales |
| 04 | PY | imports pandas, numpy, fastai, datasets, sklearn |
| 05 | MD | Conclusión: Imports listos |
| 06 | MD | Introducción: Carga del dataset |
| 07 | PY | `load_dataset("SetFit/toxic_conversations")` |
| 08 | MD | Conclusión: Dataset descargado |
| 09 | MD | Introducción: Análisis del desbalanceo |
| 10 | PY | value_counts + gráfico de barras |
| 11 | MD | Conclusión: Desbalanceo confirmado |
| 12 | MD | Introducción: Downsampling |
| 13 | PY | train_test_split estratificado + reducción clase mayoritaria al 120% |
| 14 | MD | Conclusión: Dataset equilibrado |
| 15 | MD | Introducción: Preparación para FastAI |
| 16 | PY | columna `set` para FastAI (True=validación, False=train) |
| 17 | MD | Introducción: Traductor ES→EN *(2 pts)* |
| 18 | PY | `MarianMTModel` + `MarianTokenizer` (Helsinki-NLP/opus-mt-es-en) |
| 19 | MD | Introducción: Función de traducción y prueba |
| 20 | PY | `def traducir_a_ingles()` + prueba con 4 frases |
| 21 | MD | Introducción: Embedding general *(1 pt)* |
| 22 | MD | Conclusión: Traducción verificada [COMPLETAR] |
| 23 | PY | `api.load("glove-twitter-25")` + tokenizador NLTK |
| 24 | PY | prueba similitudes + `def frase_a_vector_general()` |
| 25 | MD | Introducción: Embedding propio *(1 pt)* |
| 26 | MD | Conclusión: Embedding general validado [COMPLETAR] |
| 27 | PY | `Word2Vec` entrenado con `train_df["text"]` + `def frase_a_vector_propio()` |
| 28 | PY | prueba palabras similares del embedding propio |
| 29 | MD | Introducción: Clasificador clásico *(2 pts)* |
| 30 | MD | Conclusión: Embedding propio entrenado [COMPLETAR] |
| 31 | PY | `RandomForestClassifier` + vectorización con media de embeddings |
| 32 | PY | evaluación: `classification_report` + matriz de confusión |
| 33 | MD | Introducción: Clasificador Deep Learning *(2 pts)* |
| 34 | MD | Conclusión: Random Forest evaluado [COMPLETAR] |
| 35 | PY | `DataBlock` + `text_classifier_learner` + `AWD_LSTM` |
| 36 | PY | `lr_find()` |
| 37 | PY | `fine_tune(8, 1e-3, freeze_epochs=2)` |
| 38 | PY | evaluación test + matriz de confusión |
| 39 | MD | Introducción: Clasificador Modelo del Lenguaje *(2 pts)* |
| 40 | MD | Conclusión: AWD-LSTM completado [COMPLETAR] |
| 41 | PY | `DataBlock` con `is_lm=True` + `language_model_learner` |
| 42 | PY | `lr_find()` del language model |
| 43 | PY | `fine_tune()` del LM + `save_encoder("encoder_toxicidad")` |
| 44 | PY | clasificador con `load_encoder("encoder_toxicidad")` |
| 45 | PY | `fine_tune()` del clasificador final |
| 46 | PY | evaluación test + matriz de confusión |
| 47 | MD | Conclusión: LM+Encoder evaluado [COMPLETAR] |
| 48 | MD | Introducción: Comparativa |
| 49 | PY | gráfico de barras con los 3 accuracies |
| 50 | MD | Conclusión: Comparativa [COMPLETAR] |
| 51 | MD | Introducción: Bot administrador |
| 52 | PY | `def bot_administrador(comentario_es, umbral=0.6)` |
| 53 | MD | Conclusión: Bot operativo [COMPLETAR] |

---

## 5. Variables clave que deben existir en memoria

Estas variables se usan en celdas posteriores. Si una celda falla con `NameError`, significa que una celda anterior no se ejecutó correctamente.

```python
random_seed = 33
df                  # DataFrame con todos los datos (train+test unidos)
train_df            # DataFrame de train tras downsampling
test_df             # DataFrame de test
train_df_fastai     # train_df con columna 'set' para FastAI
tokenizer_trad      # MarianTokenizer cargado
modelo_traduccion   # MarianMTModel cargado
embedding_general   # modelo GloVe cargado con gensim
tokenizer_nltk      # WordPunctTokenizer()
embedding_propio    # Word2Vec entrenado con train_df
clf_clasico         # RandomForestClassifier entrenado
acc_clasico         # float con accuracy del RF en test
learner_deep        # FastAI text_classifier_learner (AWD-LSTM)
acc_deep            # float con accuracy del AWD-LSTM en test
language_model      # FastAI language_model_learner
learner_lm          # FastAI text_classifier_learner con encoder
acc_lm              # float con accuracy del LM+Encoder en test
```

---

## 6. Regla anti-leakage (IMPORTANTE)

El entrenamiento de Word2Vec y cualquier `fit()` se hace **exclusivamente con `train_df`**, nunca con `test_df` ni con el `df` completo. El split train/test se hace antes de construir cualquier embedding propio.

---

## 7. Cuadernillos del profesor en los que se basa la solución

| Cuadernillo | Qué aporta |
|---|---|
| `601_Embeddings` | Word2Vec con Gensim, GloVe con gensim.downloader, función media de embeddings |
| `602_Clasificador_de_textos` | DataBlock FastAI, text_classifier_learner, language_model_learner, save/load encoder |
| `605_Reducción_de_datos` | Técnica de downsampling (threshold=1.2) con train_test_split estratificado |
| `606_Aumento_de_datos` | MarianMT para traducción (Helsinki-NLP/opus-mt-es-en) |

El código del notebook sigue los mismos patrones, nombres de variables y estructura que estos cuadernillos. Cuando hay dudas de implementación, la referencia son esos ejemplos.
