# AGENTS.md

## Cursor Cloud specific instructions

This is a Python research project for a Legal-Domain RAG system (Korean real estate tax law). There is no web application, no Docker, no build system, and no formal test framework. The project uses pure Python scripts and Jupyter notebooks.

### Environment

- Python 3.12 with a virtualenv at `.venv`. Activate with `source /workspace/.venv/bin/activate`.
- All Python dependencies are listed in `README.md` (§ 사용법 > 환경 준비). There is no `requirements.txt` or `pyproject.toml`.
- **LangChain version constraint**: This codebase uses `from langchain.retrievers import EnsembleRetriever` which requires `langchain<1.0`. The update script pins `"langchain>=0.3,<1.0"` and matching `langchain-openai`, `langchain-community`, `langchain-core` versions. Do not upgrade to langchain 1.x without updating all import paths.
- `OPENAI_API_KEY` is **required** for any RAG pipeline execution (LLM generation). Without it, only document retrieval / reranker initialization tests work.
- `COHERE_API_KEY` is optional (only for `RAG_Cohere_Rerank_FINAL.py`).
- LangSmith tracing is enabled by default but non-blocking — warnings about missing `LANGSMITH_API_KEY` can be ignored safely.

### Running tests

- **Import verification**: `python RAG_Retriever_Reranker_Experiment/test_imports.py` — references non-FINAL module names, so most will fail. This is a pre-existing issue.
- **Working test scripts** (use `_FINAL` module names):
  - `cd RAG_Retriever_Reranker_Experiment && python final_test.py` — BM25 + CrossEncoder + Embedding reranker smoke test
  - `cd RAG_Retriever_Reranker_Experiment && python test_final_rerankers.py` — more thorough FINAL reranker test
- BM25 and CrossEncoder tests load documents and perform retrieval without needing `OPENAI_API_KEY`. Embedding reranker initialization requires the key.
- There is no linter, formatter, or type checker configured for this project.

### Key data files

- `Naive_RAG/output_chunks_with_embeddings.json` (~34 MB) and `RAG_Retriever_Reranker_Experiment/output_chunks_with_embeddings.json` are pre-built embedding files already in the repo.
- `Q-A Data for Ragas Evaluation/real_estate_tax_QA.json` is the ground-truth QA dataset.

### Directory notes

- Directory and file names contain spaces and Korean characters (e.g., `BGE 계열/`, `Case Data Crawling/`). Always quote paths.
- `RAG_with_Various_Rerankers/` subdirectories have no `__init__.py` but work via Python 3 namespace packages since the parent is added to `sys.path`.
