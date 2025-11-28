import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()

EMBEDDING_MODEL = 'text-embedding-3-large'
CHAT_MODEL = 'gpt-4.1-mini'
TOP_K = 5

SYSTEM_PROMPT = (
    'You are a strict Punjabi/English RAG assistant. '
    'Answer only from the provided context. '
    'If the answer is not available, say "The information is not present in the provided text." '
    'Cite chunk IDs when referencing the source.'
)

app = Flask(__name__)


def load_chunks(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f'{path} not found, please run build_index.py first.')
    return json.loads(path.read_text(encoding='utf-8'))


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise SystemExit(f'{path} not found, please run build_index.py first.')
    return faiss.read_index(str(path))


def ensure_index_assets() -> None:
    index_path = Path('index.faiss')
    chunks_path = Path('chunks.json')
    if index_path.exists() and chunks_path.exists():
        return
    print('Index or chunk list missing; running build_index.py before start.')
    subprocess.run(['python3', 'build_index.py'], check=True)


def embed_query(client: OpenAI, question: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    vector = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector


def retrieve_context(
    client: OpenAI,
    index: faiss.Index,
    chunks: List[dict],
    question: str,
) -> List[dict]:
    vector = embed_query(client, question)
    _, indices = index.search(vector, TOP_K)
    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx])
    return retrieved


def format_context(chunks: List[dict]) -> str:
    pieces = []
    for chunk in chunks:
        chunk_id = chunk.get('id', 'unknown')
        samples = chunk['text'].strip()
        pieces.append(f'[{chunk_id}] {samples}')
    return '\n\n'.join(pieces)


def ask_question(question: str, client: OpenAI, index: faiss.Index, chunks: List[dict]) -> Tuple[str, List[dict]]:
    question = question.strip()
    if not question:
        return 'Please ask a question.', []

    retrieved = retrieve_context(client, index, chunks, question)
    if not retrieved:
        return 'No context could be retrieved from the index.', []

    context = format_context(retrieved)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': (
                'Context:\n'
                f'{context}\n\n'
                'Answer the user question strictly from the context above. '
                'Reference the chunk IDs when citing facts. '
                'If nothing relevant is present, reply with "The information is not present in the provided text."\n\n'
                f'Question: {question}'
            ),
        },
    ]

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0)
    answer = resp.choices[0].message.content.strip()
    return answer, retrieved


def ask_loop(client: OpenAI, index: faiss.Index, chunks: List[dict]) -> None:
    print('Chatbot ready. Ask a question (or type q to quit).')
    while True:
        question = input('\nQuestion: ').strip()
        if not question or question.lower() in {'q', 'quit', 'exit'}:
            print('Goodbye!')
            break

        answer, _ = ask_question(question, client, index, chunks)
        print('\nAnswer:\n', answer)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    answer = ''
    question = ''
    chunks_html = ''
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        answer, retrieved = ask_question(question, app.config['client'], app.config['index'], app.config['chunks'])
        if retrieved:
            chunk_lines = [
                f"<strong>Chunk {c['id']}:</strong> {c['text'][ : 300]}..."
                for c in retrieved
            ]
            chunks_html = '<br>'.join(chunk_lines)
    return render_template_string(
        '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Gurbani RAG Chatbot</title>
            <style>
                body { font-family: system-ui, sans-serif; margin: 2rem; background:#111; color:#f5f5f5; }
                textarea, input { width: 100%; font-size: 1rem; padding: 0.5rem; }
                button { font-size: 1rem; padding: 0.75rem 1.25rem; margin-top:0.5rem; }
                .card { background:#1b1b1b; border:1px solid #333; padding:1rem; margin-top:1rem; border-radius:0.75rem; }
                .chunks { max-height:200px; overflow:auto; border:1px dashed #555; padding:0.5rem; }
            </style>
        </head>
        <body>
            <h1>Punjabi RAG Chatbot</h1>
            <form method="post">
                <label for="question">Ask your question (English or Punjabi):</label>
                <textarea id="question" name="question" rows="3">{{ question }}</textarea>
                <button type="submit">Ask</button>
            </form>
            {% if answer %}
            <div class="card">
                <h2>Answer</h2>
                <p>{{ answer }}</p>
            </div>
            {% endif %}
            {% if chunks %}
            <div class="card">
                <h3>Top-k chunks</h3>
                <div class="chunks">
                    {{ chunks|safe }}
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        ''',
        answer=answer,
        question=question,
        chunks=chunks_html,
    )


def load_resources() -> Tuple[OpenAI, faiss.Index, List[dict]]:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('OPENAI_API_KEY is required in .env')

    ensure_index_assets()
    client = OpenAI(api_key=api_key)
    chunks = load_chunks(Path('chunks.json'))
    index = load_index(Path('index.faiss'))
    return client, index, chunks


def run_server():
    client, index, chunks = load_resources()
    app.config['client'] = client
    app.config['index'] = index
    app.config['chunks'] = chunks
    print('Serving chatbot on http://127.0.0.1:7860')
    app.run(host='127.0.0.1', port=7860)


def main():
    parser = argparse.ArgumentParser(description='Ask GPT via CLI or localhost web UI.')
    parser.add_argument('--cli', action='store_true', help='Run the interactive console UI instead of the web server.')
    parser.add_argument('--check', action='store_true', help='Validate resources and exit.')
    args = parser.parse_args()

    client, index, chunks = load_resources()
    if args.check:
        print('Resources loaded; index contains', len(chunks), 'chunks.')
        return

    if args.cli:
        ask_loop(client, index, chunks)
    else:
        app.config['client'] = client
        app.config['index'] = index
        app.config['chunks'] = chunks
        print('Serving chatbot on http://127.0.0.1:7860')
        app.run(host='127.0.0.1', port=7860)


if __name__ == '__main__':
    main()
