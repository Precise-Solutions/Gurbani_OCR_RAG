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
            chunk_lines = []
            for c in retrieved:
                snippet = c['text'][:320].strip()
                chunk_lines.append(
                    f"<div class='chunk'><strong>Chunk {c['id']}</strong>{snippet}...</div>"
                )
            chunks_html = ''.join(chunk_lines)
    return render_template_string(
        '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Gurbani RAG Chatbot</title>
            <style>
                :root {
                    color-scheme: dark;
                }
                * { box-sizing: border-box; }
                body {
                    background: linear-gradient(180deg, #050505, #1d1d1f);
                    font-family: 'Inter', system-ui, sans-serif;
                    color: #f8fafc;
                    margin: 0;
                    min-height: 100vh;
                }
                .page {
                    padding: 2rem;
                    max-width: 1100px;
                    margin: 0 auto;
                    display: flex;
                    flex-direction: column;
                    gap: 1.75rem;
                }
                .hero {
                    margin-bottom: 1.5rem;
                }
                h1 {
                    font-size: 2.25rem;
                    margin-bottom: 0.25rem;
                }
                .subtitle {
                    color: #cbd5f5;
                    margin-top: 0;
                }
                form {
                    margin-bottom: 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.85rem;
                }
                textarea {
                    width: 100%;
                    min-height: 120px;
                    border-radius: 0.75rem;
                    border: 1px solid #2d3748;
                    background: #0f172a;
                    color: #f1f5f9;
                    padding: 0.9rem;
                    font-size: 1rem;
                    resize: vertical;
                    font-family: inherit;
                }
                button {
                    background: #2563eb;
                    border: none;
                    color: white;
                    padding: 0.85rem 1.75rem;
                    border-radius: 0.75rem;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                    align-self: flex-start;
                }
                button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4);
                }
                .grid {
                    display: grid;
                    gap: 1.75rem;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                }
                .card {
                    background: rgba(15, 23, 42, 0.9);
                    border: 1px solid rgba(148, 163, 184, 0.3);
                    border-radius: 1rem;
                    padding: 1.25rem;
                    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.65);
                }
                .card h2, .card h3 {
                    margin-top: 0;
                    margin-bottom: 0.75rem;
                }
                .answer {
                    line-height: 1.6;
                    max-height: 380px;
                    overflow: auto;
                }
                .chunks {
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                    max-height: 340px;
                    overflow: auto;
                    padding-right: 0.25rem;
                }
                .chunk {
                    border-radius: 0.75rem;
                    padding: 0.85rem;
                    border: 1px dashed rgba(148, 163, 184, 0.5);
                    background: rgba(255, 255, 255, 0.02);
                    font-size: 0.95rem;
                    line-height: 1.4;
                }
                .chunk strong {
                    font-size: 0.95rem;
                    display: block;
                    margin-bottom: 0.35rem;
                }
                .info {
                    font-size: 0.9rem;
                    color: #94a3b8;
                    margin-bottom: 1rem;
                }
                @media (max-width: 640px) {
                    .page {
                        padding: 1.25rem;
                    }
                    textarea {
                        min-height: 100px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="page">
                <div class="hero">
                <h1>Gurbani RAG AI Chatbot</h1>
                <p class="subtitle">Ask questions in English or Punjabi. Answers are grounded in the uploaded OCR text.</p>
                </div>
                <form method="post">
                    <label for="question">Ask your question:</label>
                    <textarea id="question" name="question" rows="4" placeholder="Type something like: 'Who does the story mention?'">{{ question }}</textarea>
                    <button type="submit">Get Answer</button>
                    <p class="info">Responses cite top chunks retrieved from the OCR text.</p>
                </form>
                <div class="grid">
                    {% if answer %}
                    <div class="card">
                        <h2>Answer</h2>
                        <div class="answer">{{ answer }}</div>
                    </div>
                    {% endif %}
                    {% if chunks %}
                    <div class="card">
                        <h3>Top-k Chunks</h3>
                        <div class="chunks">
                            {{ chunks|safe }}
                        </div>
                    </div>
                    {% endif %}
                </div>
            </div>
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
    port = int(os.getenv('PORT', '7860'))
    print('Serving chatbot on http://0.0.0.0:%s' % port)
    app.run(host='0.0.0.0', port=port)


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
        port = int(os.getenv('PORT', '7860'))
        print('Serving chatbot on http://0.0.0.0:%s' % port)
        app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
