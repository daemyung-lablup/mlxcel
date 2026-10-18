# 05. 내 문서로 답하는 RAG

폴더의 Markdown 문서를 조각내 임베딩하고, 질문과 가까운 조각을 찾아 리랭커로 다시 고른 뒤, 그 내용만 근거로 답하게 합니다. **채팅 모델, 임베딩 모델, 리랭커를 서버 하나에** 올립니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 문서 검색 → 리랭크 → 근거 기반 답변, 출처 표시 |
| 모델 | 채팅 `mlx-community/Qwen3-4B-4bit`, 임베딩 `Qwen/Qwen3-Embedding-0.6B` (1.1 GB), 리랭커 `mlx-community/Qwen3-Reranker-0.6B-4bit` (331 MB) |
| 메모리 | 16 GB 이상 |
| 배우는 것 | `/v1/embeddings`, `/v1/rerank`, 사이드 모델 서빙 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18. 지식 베이스로 mlxcel 저장소의 `docs/` (Markdown 30개)를 썼습니다.

## 1. 서버: 세 모델을 함께

```bash
mlxcel download Qwen/Qwen3-Embedding-0.6B
mlxcel download mlx-community/Qwen3-Reranker-0.6B-4bit

mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --port 8080
```

로그에서 세 모델이 모두 준비됐는지 확인합니다.

```text
Embedding model ready on /v1/embeddings model_type=Qwen3Embedding dim=1024 max_length=8192
Reranker ready on /v1/rerank kind="generative_text" model_type=qwen3 max_length=8192
```

## 2. 코드

```python
# /// script
# dependencies = ["openai>=1.60", "httpx", "numpy"]
# ///
import pathlib
import sys
import httpx
import numpy as np
from openai import OpenAI

BASE = "http://localhost:8080"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")
DOCS = pathlib.Path(sys.argv[1])


# 1. Split every Markdown file into chunks at headings (about 1,500 chars max).
def chunks(path):
    part = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and part or sum(map(len, part)) > 1500:
            yield "\n".join(part)
            part = []
        part.append(line)
    if part:
        yield "\n".join(part)


passages = [(p.name, c) for p in sorted(DOCS.glob("*.md")) for c in chunks(p) if len(c) > 200]
print(f"indexed {len(passages)} chunks from {len(list(DOCS.glob('*.md')))} files")


# 2. Embed all chunks once (batched), normalise for cosine similarity.
def embed(texts, instruction=None):
    vecs = []
    for i in range(0, len(texts), 32):
        body = {"input": texts[i:i + 32]}
        if instruction:
            body["instruction"] = instruction
        data = httpx.post(f"{BASE}/v1/embeddings", json=body, timeout=300).json()["data"]
        vecs += [d["embedding"] for d in data]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


index = embed([c for _, c in passages])


def answer(question):
    # 3. Retrieve: cosine similarity, keep the top 10 candidates.
    q = embed([question], instruction="Given a question, retrieve passages that answer it")[0]
    top = np.argsort(-(index @ q))[:10]
    # 4. Rerank the candidates with the reranker, keep 3.
    ranked = httpx.post(f"{BASE}/v1/rerank", json={
        "query": question, "documents": [passages[i][1] for i in top], "top_n": 3,
    }, timeout=300).json()["results"]
    picked = [passages[top[r["index"]]] for r in ranked]
    for r, (name, _) in zip(ranked, picked):
        print(f"  [{r['relevance_score']:.3f}] {name}")
    # 5. Generate an answer grounded in the picked passages.
    context = "\n\n".join(f"[{n}]\n{c}" for n, c in picked)
    r = client.chat.completions.create(
        model="qwen3-4b", max_tokens=400,
        messages=[{"role": "system", "content":
                   "Answer only from the context. Cite file names in brackets. "
                   "If the context does not contain the answer, say so."},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content


for question in ["How do I serve an embedding model and a reranker next to a chat model?",
                 "Whisper 서버는 어떤 오디오 형식을 받나요?",
                 "What is the capital of France?"]:
    print(f"\nQ: {question}")
    print(f"A: {answer(question)}")
```

```bash
uv run rag.py path/to/mlxcel/docs
```

## 실행 결과

```text
indexed 890 chunks from 30 files

Q: How do I serve an embedding model and a reranker next to a chat model?
  [0.999] embeddings.md
  [0.996] environment-variables.md
  [0.990] embeddings.md
A: ... you can use the `--embedding-model` and `--reranker-model` flags when starting the
   server ... See [environment-variables.md] for more details.

Q: Whisper 서버는 어떤 오디오 형식을 받나요?
  [0.985] audio-api.md
  [0.967] llama-server-compat.md
  [0.899] audio-api.md
A: Whisper 서버는 WAV만을 받습니다. ... MP3와 FLAC는 Phi-4 Multimodal 및 Gemma 3n
   chat-model 경로에서 지원되지만 ... [audio-api.md]

Q: What is the capital of France?
  [0.003] environment-variables.md
  [0.001] responses-api.md
  [0.000] embeddings.md
A: The context provided does not contain information about the capital of France. ...
```

세 번째 질문처럼 문서에 없는 내용은 **리랭크 점수가 0에 가깝게** 떨어지고, 모델도 답이 없다고 말합니다. 실제 서비스에서는 점수가 기준(예: 0.1) 아래면 생성 단계를 건너뛰면 됩니다.

## 주의

- 첫 답변 중간에 중국어 단어(`指定`)가 한 번 섞였습니다. 작은 Qwen 모델에서 가끔 나오는 현상이라, 중요한 서비스라면 더 큰 모델을 쓰거나 후처리를 넣으세요.
- `/v1/embeddings`의 `instruction` 필드는 Qwen3-Embedding용 쿼리 지시문입니다. 문서 쪽에는 넣지 않고 질문 쪽에만 넣습니다.
- `model` 필드를 보낼 경우 서버가 띄운 임베딩·리랭커 이름과 같아야 합니다. 생략해도 됩니다.

## 배운 것

- **2단계 검색**: 임베딩(빠르고 넓게) → 리랭커(느리지만 정확하게). 890개 조각에서 10개를 고르고 다시 3개로 줄였습니다.
- 사이드 모델은 각자 작업 스레드를 갖고 채팅 모델 옆에서 돕니다.

## 응용

- 사내 위키, 회의록, 코드 저장소의 README 모음에 그대로 적용
- 인덱스를 파일(`np.save`)로 저장해 두면 다음부터 임베딩 단계를 건너뜁니다.
- 답변을 [04번](04-structured-extraction.md)처럼 `{answer, sources[]}` JSON으로 받기
