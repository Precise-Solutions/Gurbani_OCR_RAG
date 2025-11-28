import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def load_source_text() -> str:
    candidate_names = ['gurbani.txt', 'Gurbani.txt']

    for name in candidate_names:
        candidate = Path(name)
        if candidate.exists():
            content = candidate.read_text(encoding='utf-8').strip()
            if content:
                return content

    raise SystemExit(
        'No OCR source found. Add gurbani.txt (or Gurbani.txt) before running this script.'
    )


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 50) -> List[str]:
    words = text.replace('\n', ' ').split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        if end == len(words):
            break
        start += step
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(model=model, input=text)
        embedding = np.array(resp.data[0].embedding, dtype='float32')
        embeddings.append(embedding)
    matrix = np.vstack(embeddings)
    faiss.normalize_L2(matrix)
    return matrix


def main() -> None:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('OPENAI_API_KEY is required in .env')

    raw = load_source_text()
    chunks = chunk_text(raw)

    if not chunks:
        raise SystemExit('No text to index; check gurbani.txt or gurbani_cleaned.txt')

    print(f'Preparing {len(chunks)} chunks for indexing.')

    client = OpenAI(api_key=api_key)
    model = 'text-embedding-3-large'
    matrix = embed_texts(client, chunks, model)

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, 'index.faiss')

    chunk_payload = [
        {'id': idx, 'text': chunk}
        for idx, chunk in enumerate(chunks, start=1)
    ]
    Path('chunks.json').write_text(json.dumps(chunk_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Index built with', len(chunks), 'chunks.')


if __name__ == '__main__':
    main()
