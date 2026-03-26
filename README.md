# PIA UD6 — Clasificador de Toxicidad en Redes Sociales

Repositorio de contexto para el agente de Gemini en Google Colab.

## Cómo usar este repositorio

Al inicio de cada sesión en Colab, clona este repositorio para cargar todos los documentos de contexto:

```python
!git clone https://github.com/TU_USUARIO/PIA_UD6_contexto.git
```

Luego carga los documentos en el chat del agente:

```
Lee los siguientes documentos en orden antes de ayudarme:
1. PIA_UD6_contexto/01_TAREA_Y_SOLUCION.md
2. PIA_UD6_contexto/02_GUIA_AGENTE.md
3. PIA_UD6_contexto/03_GUIA_DEPURACION.md

A partir de ahora actúa según las instrucciones de esos documentos.
```

## Documentos del repositorio

| Archivo | Contenido |
|---|---|
| `01_TAREA_Y_SOLUCION.md` | Enunciado, rúbrica, dataset, estructura del notebook y variables clave |
| `02_GUIA_AGENTE.md` | Cómo debe actuar el agente: tono, estilo, flujo de trabajo |
| `03_GUIA_DEPURACION.md` | Errores comunes y sus fixes |
| `PIA_06_Tarea_Alfredo_FINAL.ipynb` | El notebook principal (54 celdas) |

## Notebook

El notebook `PIA_06_Tarea_Alfredo_FINAL.ipynb` está listo para ejecutar en Google Colab con GPU T4.

Antes de ejecutar: `Entorno de ejecución → Cambiar tipo → GPU T4`

## Tarea

**Módulo:** Programación de Inteligencia Artificial  
**Unidad:** 6 — Procesamiento del Lenguaje Natural  
**Objetivo:** Clasificador de toxicidad para comentarios de redes sociales en español
