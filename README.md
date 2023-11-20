# Production RAG for Arabic Legal & Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Retrieval-Augmented Generation (RAG) platform specialized for Arabic legal documents, contracts, and regulatory filings. Built with a hybrid retrieval pipeline combining dense semantic search (Multilingual-E5 / Qdrant) with lexical search (BM25), cross-encoder re-ranking, and strict hallucination mitigation.

---

## 🏛️ Architecture Overview

```
                      +---------------------------------------+
                      |           Client Request              |
                      |   (Legal Query / Contract Question)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Arabic Text Normalizer          |
                      |  (Tashkeel, Tatweel, Hamza Unification)|
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
         +--------------------------+           +--------------------------+
         |    Lexical Retrieval     |           |     Dense Retrieval      |
         |    (BM25 with Tokenizer) |           | (Multilingual-E5/Qdrant) |
         +------------+-------------+           +-------------+------------+
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Reciprocal Rank Fusion (RRF)      |
                      |          (Candidate Pool)             |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        Cross-Encoder Re-Ranker        |
                      |  (Fine-grained Passage Re-scoring)    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Constrained Context Builder      |
                      |    + Anti-Hallucination Prompting     |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       LLM Generation & Citations      |
                      |  (LLaMA 3 / GPT-4 / Arabic Fine-tuned)|
                      +-------------------+-------------------+
```

---

## ✨ Key Features

- **Specialized Arabic Preprocessing**: Handles Arabic diacritics (Tashkeel removal), Kashida/Tatweel normalization, and Alef/Yaa/Taa Marbuta unification.
- **Hybrid Search (BM25 + Dense)**: Solves the vocabulary mismatch and legal jargon recall problem by combining BM25 keyword matching with dense embeddings (`multilingual-e5-base`).
- **Reciprocal Rank Fusion (RRF)**: Merges heterogeneous scoring spaces into an optimal rank distribution.
- **Cross-Encoder Re-Ranking**: Filters out false positives and scores the top-K chunks for semantic precision.
- **Citation & Provenance Tracking**: Every returned answer includes exact paragraph IDs, document metadata, and confidence scores.
- **Production-Ready Microservice**: FastAPI server with async endpoints, P95 latency < 900ms, Prometheus metrics, and Docker containerization.
- **Benchmark Suite**: Included evaluation dataset and metrics computation (`Hit@5`, `Hit@10`, `MRR`, `NDCG@10`).

---

## 📊 Benchmark Results (Arabic Legal 500-Query Gold Standard)

| Pipeline Configuration | Hit@5 (%) | Hit@10 (%) | MRR@10 | P95 Latency (ms) |
|:---|:---:|:---:|:---:|:---:|
| Lexical BM25 (Baseline) | 68.4 | 74.2 | 0.612 | **120** |
| Dense Embedding Only (E5) | 76.1 | 82.5 | 0.704 | 240 |
| Hybrid (BM25 + E5 with RRF) | 83.2 | 89.1 | 0.781 | 310 |
| **Hybrid + Cross-Encoder (Ours)** | **88.6** | **93.8** | **0.849** | **780** |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yousefkhaled2005/production-rag-arabic.git
cd production-rag-arabic

python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run with Docker Compose

To spin up both Qdrant Vector DB and the FastAPI RAG service:

```bash
docker compose up -d --build
```

The API will be accessible at `http://localhost:8000` and Swagger docs at `http://localhost:8000/docs`.

---

## 🛠️ API Reference

### Health Check
`GET /health`

### Query the Legal Knowledge Base
`POST /api/v1/query`

**Request:**
```json
{
  "query": "ما هي شروط فسخ العقد محدد المدة في قانون العمل المصري؟",
  "top_k": 5,
  "use_reranker": true,
  "temperature": 0.1
}
```

**Response:**
```json
{
  "answer": "وفقاً للمادة (104) من قانون العمل رقم 12 لسنة 2003، ينتهي عقد العمل محدد المدة بانقضاء مدته...",
  "citations": [
    {
      "document_id": "egypt_labor_law_2003",
      "chunk_id": "chunk_104_1",
      "relevance_score": 0.942,
      "text": "المادة 104: ينتهي عقد العمل محدد المدة بانقضاء مدته، فإذا أبرم العقد لمدة تزيد على خمس سنوات..."
    }
  ],
  "latency_ms": 745.2
}
```

### Ingest Documents
`POST /api/v1/documents/index`

---

## 🧪 Running Tests & Evaluation

```bash
# Run unit & integration tests
pytest tests/ -v

# Run the benchmark evaluation
python -m evaluation.benchmark
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## 👤 Author

**Yousef Khaled**  
Senior AI Systems Architect  
- GitHub: [@yousefkhaled2005](https://github.com/yousefkhaled2005)
