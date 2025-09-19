# Financial Document Q&A Assistant

## Overview
A local Streamlit app to upload financial PDFs and Excel files and ask natural-language questions about revenue, expenses, profits, etc. The app extracts text and tables, creates a retrieval index using TF-IDF, and uses a local Ollama small language model to generate answers from the retrieved context.

## Requirements
- Python 3.9+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Install Ollama locally (https://ollama.ai/) and pull or create a model:
  ```
  # example (depends on model availability)
  ollama pull llama2
  ```
  Replace `llama2` with your local model name.

## Run
1. Start Streamlit:
   ```
   streamlit run streamlit_app.py
   ```
2. In the sidebar set the `Local Ollama model name` to your model (default `llama2`).
3. Upload PDF and/or Excel files.
4. Click "Process uploaded files".
5. Ask questions in the chat box.

## Notes and Tips
- The app uses the Ollama CLI (`ollama generate <model> "<prompt>"`). If you prefer HTTP, update `qa.call_ollama_cli()` accordingly.
- The retrieval is TF-IDF based (fast & simple); for larger corpora consider sentence embeddings + approximate nearest neighbour (e.g., FAISS).
- PDF table extraction uses `pdfplumber`. Table extraction is heuristic — for complex statements (multi-column layouts, images), pre-processing or OCR may be needed.
- For production, sanitize prompts, add rate-limiting, and protect sensitive documents.

## Limitations & Extensions
- Does not use embeddings currently — TF-IDF works well for short documents but embeddings + FAISS will scale better.
- You can add a numeric extractor to answer direct numeric queries without invoking the LLM (fast path).
- Add conversational memory: pass previous Q/A into the prompt for follow-ups.
