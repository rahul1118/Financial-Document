# document_processor.py
import io
import re
from typing import List, Tuple, Dict
import pandas as pd
import pdfplumber

def extract_text_and_tables_from_pdf(file_stream: io.BytesIO) -> Dict:
    """
    Returns dict with keys:
      - text_blocks: list[str] (paragraph-like chunks from PDF text)
      - tables: list[pandas.DataFrame]
      - metadata: minimal metadata
    """
    text_blocks = []
    tables = []
    metadata = {}
    file_stream.seek(0)
    with pdfplumber.open(file_stream) as pdf:
        metadata['pages'] = len(pdf.pages)
        for page in pdf.pages:
            # extract text
            text = page.extract_text() or ""
            # split heuristically into paragraphs / lines and keep meaningful lines
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if lines:
                # join small groups to make blocks
                combined = "\n".join(lines)
                text_blocks.append(combined)
            # extract tables
            page_tables = page.extract_tables()
            for tbl in page_tables:
                try:
                    # convert to DataFrame (first row may be header)
                    df = pd.DataFrame(tbl)
                    # try to promote header if it seems headers exist
                    # if first row has text and subsequent rows are numeric-ish, promote
                    tables.append(df)
                except Exception:
                    pass
    return {"text_blocks": text_blocks, "tables": tables, "metadata": metadata}

def extract_text_and_tables_from_excel(file_stream: io.BytesIO) -> Dict:
    file_stream.seek(0)
    # pandas can read from BytesIO
    xls = pd.read_excel(file_stream, sheet_name=None, header=None)
    text_blocks = []
    tables = []
    metadata = {"sheets": list(xls.keys())}
    for sheet_name, df in xls.items():
        # store sheet content as DataFrame
        tables.append(df)
        # also a textual representation: headings + first N rows
        try:
            txt = f"Sheet: {sheet_name}\n" + df.head(50).to_string(index=False)
            text_blocks.append(txt)
        except Exception:
            pass
    return {"text_blocks": text_blocks, "tables": tables, "metadata": metadata}

def normalize_numeric_tokens(text: str) -> str:
    # simple helper to normalize common number formats (commas) to raw numbers
    return re.sub(r'(?<=\d),(?=\d)', '', text)

def chunk_text_blocks(text_blocks: List[str], max_chars=800, overlap=200) -> List[str]:
    """
    Naive chunker: splits each block by max_chars with overlap
    """
    chunks = []
    for block in text_blocks:
        block = block.strip()
        if not block:
            continue
        block = normalize_numeric_tokens(block)
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        start = 0
        L = len(block)
        while start < L:
            end = start + max_chars
            chunk = block[start:end]
            chunks.append(chunk.strip())
            start = max(0, end - overlap)
    return chunks

def tables_to_text_snippets(tables: List[pd.DataFrame], max_cells=200) -> List[str]:
    """
    Converts DataFrame tables into readable textual snippets.
    """
    snippets = []
    for idx, df in enumerate(tables):
        try:
            small = df.copy()
            # convert to string representation (first rows)
            txt = f"Table {idx}:\n" + small.head(50).to_string(index=False)
            snippets.append(txt)
        except Exception:
            continue
    return snippets
