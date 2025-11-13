# Smart University Document Assistant

Telegram bot for working with university documents using OpenAI API. The bot can summarize documents, create embeddings for fast search, and generate answers to questions based on uploaded documents.

## Features

- **Document Upload**: Upload text files (TXT) and PDF files with lectures, policies, notes
- **Automatic Summarization**: Bot creates brief summaries of documents
- **Smart Search**: Fast search for relevant information by queries
- **Question Answering**: Get accurate answers based on uploaded documents
- **Vector Search**: Uses FAISS for efficient embedding-based search
- **Query Routing (Modular RAG)**: Automatic selection of optimal search strategy
- **Answer Caching**: Repeated queries are processed instantly
- **Conversation History**: Bot remembers context of previous messages
- **Automated Testing**: Benchmarking system for RAG quality assessment
- **Retry Mechanism**: Automatic retries on API failures

## Installation

1. Clone the repository or download files

2. **Update pip, setuptools and wheel** (important for Windows):
```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Install dependencies:

   **Option A: With FAISS (recommended for large volumes)**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Without FAISS (if compilation issues)**
   ```bash
   pip install -r requirements_simple.txt
   ```
   Additionally, you can configure `document_store_simple.py` for your needs (import already configured in `bot_simple.py`)

   **Or use installation script for Windows:**
   ```bash
   install.bat
   ```

4. Create `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

5. Fill in environment variables:
   - `OPENAI_API_KEY` - your OpenAI API key
   - `TELEGRAM_BOT_TOKEN` - bot token from @BotFather

### Troubleshooting Installation

If you have issues installing NumPy on Windows:
1. Use precompiled wheels: `python -m pip install numpy --only-binary :all:`
2. Then install remaining dependencies: `pip install -r requirements_simple.txt`

## Running

**Check installation:**
```bash
python tests/check_installation.py
```

**Run the bot:**

```bash
python run_bot.py
```

## Usage

1. Start the bot with `/start` command
2. Send a text file (TXT), PDF file, or text message
3. Ask a question about the document
4. Get an accurate answer!

To delete a document, use the command:

```bash
/delete <ID>
```

ID can be found via `/docs`.

### Commands

- `/start` - start working with the bot
- `/help` - usage help
- `/docs` - list of your documents
- `/add_doc` - add text as document
- `/delete <ID>` - delete document (ID can be found via `/docs`)
- `/clear` - clear conversation history
- `/routing [query]` - information about query routing (Modular RAG)

## Architecture

### Project Structure

```
ForKupa/
├── src/                    # Main bot code
│   ├── bot_simple.py      # Main file with Telegram handlers
│   ├── openai_service.py  # Service for OpenAI API
│   ├── pdf_extractor.py   # PDF text extraction
│   ├── document_store.py  # Storage with FAISS (optional)
│   └── document_store_simple.py  # Storage with NumPy
│
├── utils/                  # Utility modules
│   ├── query_router.py    # Query Routing for Modular RAG
│   ├── cache_manager.py   # Answer caching with TTL
│   ├── conversation_manager.py  # Conversation history
│   ├── retry_handler.py   # Retry with exponential backoff
│   └── logger_config.py   # Logging with rotation
│
├── tests/                  # Tests and checks
│   ├── benchmark_test.py  # RAG benchmarking system
│   ├── test_dataset.json  # Test dataset
│   └── check_installation.py  # Dependency check
│
├── run_bot.py             # Bot launcher
├── requirements.txt       # Dependencies (with FAISS)
├── requirements_simple.txt  # Dependencies (without FAISS)
└── README.md
```

### Storage Version Selection

- **With FAISS** (`document_store.py`) - faster for large data volumes, requires compilation
- **Without FAISS** (`document_store_simple.py`) - easier to install, works on pure NumPy, sufficient for small volumes

### Supported Formats

- **TXT** - text files
- **PDF** - PDF documents (text extracted automatically)
- **Text** - can be sent directly via `/add_doc`

## How It Works

### Multi-Representation Indexing (from Lecture 9)

The system uses the **Multi-Representation Indexing** pattern:

1. **Document Upload**: User sends document (TXT or PDF)
2. **Text Extraction**: If PDF - text is extracted from file
3. **Summarization**: Bot creates brief summary via OpenAI (summary for retrieval)
4. **Embedding**: Vector representation of **summary** is created (not entire document!)
5. **Storage**: Full text saved in doc store, summary embedding - in vector store
6. **Search with Query Routing**:
   - Query is classified by type (factual, analytical, procedural, conceptual, comparison)
   - Optimal strategy is automatically selected (precise, broad, comprehensive)
   - Parameters configured: top_k and similarity_threshold
7. **Retrieval**: Relevant summaries found by query embedding
8. **Generation**: Full documents used to generate answer (not summaries!)

### Query Routing (Modular RAG)

System automatically analyzes each query and selects optimal parameters:

- **Factual queries** → PRECISE strategy (top_k=3, threshold=0.0)
- **Analytical/Procedural** → BROAD strategy (top_k=7, threshold=0.0)
- **Comparison queries** → COMPREHENSIVE strategy (top_k=10, threshold=0.0)

**Note:** threshold=0.0 means top_k documents with highest similarity are taken without filtering. This is optimal for Multi-Representation Indexing, where specific terms may be in full text but not in summary.

### Additional Optimizations

- **Caching**: Repeated queries returned instantly from cache
- **Conversation History**: Context of last 6 messages for coherent answers
- **Retry Mechanism**: Automatic retries on rate limits or connection errors
- **User Isolation**: Each user sees only their documents (O(1) lookup)

## RAG System Benchmarking

Project includes automated RAG testing system:

```bash
python tests/benchmark_test.py
```

### Evaluation Metrics

1. **Retrieval Metrics**:
   - Precision: proportion of relevant documents among found
   - Recall: proportion of found documents among all relevant
   - F1-Score: harmonic mean of precision and recall

2. **Answer Quality Metrics**:
   - Keyword Overlap: matching keywords with ground truth
   - Length Similarity: answer length match
   - Semantic Similarity: semantic similarity (evaluated via LLM)

### Test Dataset

`test_dataset.json` contains 10 questions covering:
- Final project requirements (lecture 8)
- Multi-Representation and other RAG techniques (lecture 9)
- Bot functionality

Each test includes:
- `question` - test question
- `expected_doc_titles` - expected relevant documents
- `ground_truth_answer` - reference answer

## Configuration

In `openai_service.py` you can change:
- `model` - generation model (default `gpt-4o-mini`)
- `embedding_model` - embedding model (default `text-embedding-3-small`)

In `.env` you can configure:
- `OPENAI_API_KEY` - OpenAI API key
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `STORAGE_DIR` - directory for document storage
- `CACHE_DIR` - directory for cache
- `CONVERSATIONS_DIR` - directory for conversation history
- `LOG_LEVEL` - logging level (INFO, DEBUG, WARNING, ERROR)

## Usage Examples

**Upload document:**
```
Send file lecture_notes.txt or document.pdf
```

**Question:**
```
What are the grading rules in the course?
```

**Answer:**
```
Bot will find relevant information and give an accurate answer based on documents.
```

## Requirements

- Python 3.8+
- OpenAI API key
- Telegram Bot Token

## License

MIT
