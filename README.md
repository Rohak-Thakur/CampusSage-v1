# CampusSage

CampusSage is a local retrieval-augmented generation (RAG) application for answering questions from campus PDFs. It extracts text from PDF documents, creates searchable text chunks, retrieves relevant content with embeddings, and sends the retrieved context to a language model.

Answers include source references in the format `[S1]`, `[S2]`, and so on. CampusSage checks whether cited source labels exist, but it does not verify the truth of every claim.

## How It Works

1. PDF text is extracted and cleaned with PyMuPDF.
2. Text is split into page-bounded chunks, with configurable overlap.
3. `sentence-transformers/all-MiniLM-L6-v2` converts chunks and questions into embeddings.
4. FAISS performs exact cosine-similarity retrieval.
5. Ollama generates answers using the retrieved chunks as context.

No paid API or API key is required. Ollama runs the language model locally on your computer.

## Requirements

- Python 3.10 or newer
- Ollama
- A local Ollama model, such as `llama3.2:3b`

## Installation

Create and activate a virtual environment:

```powershell
cd C:\path\to\CampusSage
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

## Set Up Ollama

1. Install Ollama from <https://ollama.com/download>.
2. Start the Ollama application or service.
3. Download the default free local model:

```powershell
ollama pull llama3.2:3b
```

4. Confirm that Ollama is running:

```powershell
ollama list
```

The application connects to Ollama at `http://localhost:11434` by default. Ollama settings are defined in `config.py`; for example:

```python
LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"
```

You can use another installed Ollama model with:

```powershell
python scripts/ask.py "Your question" --model <model-name>
```

For example:

```powershell
python scripts/ask.py "What are the exam instructions?" --model llama3.2:3b
```

## Prepare the Documents

Place PDF files in `data/`. The directory is ignored by Git so source documents remain local and are not included in commits.

Extract and chunk the PDFs:

```powershell
python scripts/ingest_preview.py
```

To inspect specific PDFs:

```powershell
python scripts/ingest_preview.py data/example.pdf
```

Build or rebuild the FAISS retrieval index:

```powershell
python scripts/retrieval_preview.py --rebuild
```

The generated chunk and index files are written under `data/processed/` and `data/index/`.

## Ask Questions

Ask one or more questions directly:

```powershell
python scripts/ask.py "What is the assignment deadline?"
python scripts/ask.py "What is the assignment deadline?" "How do I submit it?"
```

Start an interactive session:

```powershell
python scripts/ask.py
```

Useful options:

```powershell
python scripts/ask.py "Question" --k 6
python scripts/ask.py "Question" --show-prompt
python scripts/ask.py "Question" --model llama3.2:3b
```

You can also preview retrieval results:

```powershell
python scripts/retrieval_preview.py "What is the exam notice?"
```

## Run Tests

```powershell
pytest
```

## Project Structure

```text
data/                  Local PDF documents and generated data
scripts/
  ingest_preview.py    Extract and chunk PDFs
  retrieval_preview.py Build and test the FAISS index
  ask.py               Ask CampusSage questions
src/
  document_loader.py   PDF loading and text cleaning
  text_chunker.py      Page-bounded chunking
  embeddings.py        Sentence-transformer embedding wrapper
  retriever.py         FAISS retrieval
  prompt_templates.py  System and user prompts
  answer_generator.py  Retrieval, generation, and citation checks
  llm_engine.py        Ollama API integration
tests/                 Unit tests
config.py              Chunking, embedding, retrieval, and LLM settings
requirements.txt       Python dependencies
```

## Privacy and Data

Document processing and Ollama inference run locally. The `data/` directory is excluded through `.gitignore`, as are virtual environments, caches, environment files, editor settings, and generated Python files.