# qa.py
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

class Retriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        if not docs:
            self.vectorizer = None
            self.doc_vectors = None
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
            self.doc_vectors = self.vectorizer.fit_transform(docs)

    def top_k(self, query: str, k=3) -> List[Tuple[int, float]]:
        if self.vectorizer is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_vectors)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]

def call_ollama_cli(prompt: str, model: str = "llama2", timeout_sec: int = 30) -> str:
    """
    Calls local Ollama via the CLI: `ollama generate <model> "<prompt>"`
    Make sure Ollama is installed and the model is available locally.
    Returns the model text output or raises subprocess.CalledProcessError.
    """
    cmd = ["ollama", "generate", model, prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Install Ollama (https://ollama.ai/) and pull a model.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama call timed out.")

def make_prompt(contexts: List[str], question: str, instructions: str = None) -> str:
    header = (
        "You are a financial document assistant. Use ONLY the provided CONTEXT to answer the question. "
        "If the requested info is not in the context, say you cannot find it and suggest what could help (specific rows/figures).\n\n"
    )
    if instructions:
        header += instructions + "\n\n"
    ctx_text = "\n\n---\n\n".join(contexts) if contexts else ""
    prompt = header + "CONTEXT:\n" + ctx_text + "\n\nQUESTION:\n" + question + "\n\nAnswer succinctly and show numeric values when present. If multiple interpretations exist, list them."
    return prompt

def answer_question(retriever: Retriever, question: str, docs: List[str], model: str="llama2", top_k: int = 3) -> str:
    """
    Retrieves top_k docs and asks the model to answer.
    """
    hits = retriever.top_k(question, k=top_k)
    contexts = []
    for idx, score in hits:
        contexts.append(f"(score {score:.3f})\n{docs[idx]}")
    if not contexts:
        # fallback: send question with a small hint
        contexts = ["No relevant extracted context found. Provide the question and explain that no context exists."]
    prompt = make_prompt(contexts, question)
    try:
        response = call_ollama_cli(prompt, model=model)
    except Exception as e:
        # return a helpful error message for the UI
        response = f"ERROR calling Ollama: {str(e)}\n\nPrompt sent (truncated):\n{prompt[:2000]}"
    return response
