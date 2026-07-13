# RAG Clínico: Chunking y Embeddings para RAG Médico

Código fuente del Trabajo Fin de Máster (TFM) del Máster Universitario en Inteligencia Artificial (UNIR).

**Título**: *Optimización de Estrategias de Segmentación y Modelos de Embedding para Sistemas de Recuperación Aumentada por Generación en el Dominio Clínico*

**Autor**: José Teodosio Lorente Vallecillos

## Descripción

Sistema RAG (Retrieval-Augmented Generation) modular para consultas sobre guías de práctica clínica cardiovascular (ESC). El proyecto implementa un estudio comparativo en dos fases:

- **Fase 1**: Comparativa de estrategias de chunking (fijo, recursivo, semántico) con embedding constante
- **Fase 2**: Comparativa de modelos de embedding (2 generalistas + 2 biomédicos) con el mejor chunking de la Fase 1

## Estructura del proyecto

```
src/rag_clinico/          # Paquete principal
├── loader.py             # Carga de PDFs clínicos
├── pipeline.py           # Pipeline RAG end-to-end
├── chunking/             # 3 estrategias de segmentación
├── embedding/            # Registro de 4 modelos de embedding
├── retrieval/            # Almacén vectorial (ChromaDB)
├── generation/           # Wrapper LLM (Ollama)
└── evaluation/           # Métricas (ROUGE, BERTScore)

configs/                  # Configuración de experimentos (YAML)
data/qa_dataset/          # Dataset de evaluación (150 pares QA)
data/results/             # Resultados experimentales (JSON)
run_phase1.py             # Ejecución de Fase 1
run_phase2.py             # Ejecución de Fase 2
run_statistics.py         # Análisis estadístico (Wilcoxon)
generate_charts.py        # Generación de gráficos
```

## Stack tecnológico

- Python 3.13
- LangChain 1.2 (orquestación RAG)
- ChromaDB 1.5 (almacén vectorial)
- sentence-transformers 5.4 (embeddings locales)
- Ollama (inferencia LLM - Gemma 3 12B Q4_K_M)
- rouge-score + bert-score (métricas de evaluación)

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -e .
```

## Uso

```bash
# Fase 1: comparativa de chunking
python run_phase1.py

# Fase 2: comparativa de embeddings
python run_phase2.py

# Análisis estadístico
python run_statistics.py

# Generación de gráficos
python generate_charts.py
```

## Corpus

5 guías de práctica clínica de la European Society of Cardiology (ESC):
- Insuficiencia cardíaca (2023)
- Síndromes coronarios agudos (2023)
- Fibrilación auricular (2024)
- Hipertensión arterial (2023)
- Síndromes coronarios crónicos (2024)

Las guías no se incluyen en este repositorio por derechos de autor. Pueden descargarse desde [ESC Guidelines](https://www.escardio.org/Guidelines).

## Resultados principales

| Métrica | Mejor chunking (Fase 1) | Mejor embedding (Fase 2) |
|---------|------------------------|--------------------------|
| ROUGE-1 | Recursivo (0.298) | BGE (0.378) |
| ROUGE-L | Recursivo (0.195) | BGE (0.247) |
| BERTScore F1 | Recursivo (0.574) | BGE (0.591) |

Hallazgo principal: los modelos generalistas superan a los biomédicos en +38% ROUGE-1 (p<0.001), contradiciendo la hipótesis inicial de que los modelos especializados serían superiores en el dominio clínico.

## Licencia

MIT
