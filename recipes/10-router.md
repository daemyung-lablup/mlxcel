# 10. 한 서버에 여러 모델: 작은 모델은 분류, 큰 모델은 답변

라우터 모드로 서버 하나에 모델 여러 개를 등록해 두고, 요청의 `model` 이름으로 고릅니다. 작은 모델이 질문 난이도를 분류하고, 어려운 질문만 큰 모델이 답하게 만듭니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 분류(0.6B) → 답변(0.6B 또는 4B) 라우팅 |
| 모델 | `mlx-community/Qwen3-0.6B-4bit`, `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | `--models-preset`, 필요할 때 로드, LRU 제거 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 모델을 mlxcel 저장소에 받기

```bash
mlxcel download mlx-community/Qwen3-0.6B-4bit --force
mlxcel download mlx-community/Qwen3-4B-4bit --force
```

> ⚠️ 라우터는 **mlxcel 저장소(`~/.cache/mlxcel/models`)의 실제 파일**만 씁니다. 아래 경우는 모두 건너뛰므로 `--force`로 받아 두세요.
>
> | 시도한 방법 | 결과 |
> |---|---|
> | `mlxcel download` (HF 캐시에 이미 있음) | "reusing without re-download"로 끝남 → 라우터는 "not in the cache; download it first"라며 건너뜀 |
> | `--models-dir` 안에 HF 캐시를 가리키는 심볼릭 링크 | "resolves outside the models directory"로 건너뜀 |
> | preset의 `model =`에 HF 캐시 스냅샷 경로 | 스냅샷 파일이 링크라서 "without a regular config.json"으로 건너뜀 |

## 2. preset 파일로 이름 붙이기

`models.ini`:

```ini
[*]
ctx-size = 8192

[small]
hf-repo = mlx-community/Qwen3-0.6B-4bit
tags = fast

[large]
hf-repo = mlx-community/Qwen3-4B-4bit
tags = quality
```

`[*]`는 모든 모델의 기본값이고, 섹션 이름이 요청에서 쓸 모델 이름이 됩니다.

```bash
mlxcel-server --models-preset models.ini --models-max 2 --port 8080
```

`--models-max`는 동시에 메모리에 올려 둘 모델 수입니다. 넘치면 가장 오래 안 쓴 모델부터 내립니다(LRU).

## 3. 코드

```python
# /// script
# dependencies = ["openai>=1.60", "httpx"]
# ///
import time
import httpx
from openai import OpenAI

BASE = "http://localhost:8080"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def chat(model, prompt, max_tokens):
    t = time.perf_counter()
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens, extra_body=NO_THINK,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip(), time.perf_counter() - t


def status():
    models = httpx.get(f"{BASE}/v1/models").json()["data"]
    return {m["id"]: m["status"]["value"] for m in models if m["id"] in ("small", "large")}


def route(question):
    # 1. The small model only classifies: one word, a few tokens.
    label, t1 = chat("small", "Classify the question as SIMPLE (greeting, yes/no, one fact) "
                     f"or COMPLEX (explanation, reasoning, code). Reply with one word.\n\n{question}", 5)
    target = "large" if "COMPLEX" in label.upper() else "small"
    # 2. The chosen model answers. The router loads it on first use.
    answer, t2 = chat(target, question, 200)
    print(f"Q: {question}\n  classify[small] {label!r} {t1:.2f}s -> answer[{target}] {t2:.2f}s")
    print(f"  {answer[:160]}")


print("before:", status())
for q in ["안녕?", "Explain why a KV cache speeds up decoding, in three sentences.", "What is 2+2?"]:
    route(q)
print("after:", status())
```

## 실행 결과

```text
before: {'large': 'unloaded', 'small': 'unloaded'}
Q: 안녕?
  classify[small] 'simple' 0.61s -> answer[small] 0.19s
  안녕하세요! 궁금한 것이 있을 때마다 좋은 기도를 하시기 바랍니다. ...
Q: Explain why a KV cache speeds up decoding, in three sentences.
  classify[small] 'COMPLEX' 0.02s -> answer[large] 1.45s
  A KV cache (Key-Value cache) speeds up decoding by reusing previously computed key and value ...
Q: What is 2+2?
  classify[small] 'SIMPLE' 0.04s -> answer[small] 0.21s
  ... In standard math, 2 + 2 equals 4. ...
after: {'large': 'loaded', 'small': 'loaded'}
```

- 첫 분류 0.61초에는 `small` 모델 로드가 포함됩니다. 그다음 분류는 0.02~0.04초입니다.
- `large`는 처음 쓰일 때 자동으로 올라왔습니다(로드 + 답변 1.45초).
- 0.6B의 한국어 답("좋은 기도를 하시기 바랍니다")은 어색합니다. 작은 모델은 분류처럼 짧고 정해진 일에 쓰는 게 좋습니다.

## 관리 API

| 하고 싶은 것 | 요청 |
|---|---|
| 목록과 상태 | `GET /v1/models` (`?reload=1`이면 다시 스캔) |
| 미리 올리기 / 내리기 | `POST /models/load`, `POST /models/unload` (`{"model": "large"}`) |
| 다운로드 / 삭제 | `POST /models`, `DELETE /models?model=...` |

## 응용

- 같은 방식으로 코드 질문은 코드 모델, 이미지 질문은 VLM으로 보내기
- [09번](09-thinking.md)과 묶어 COMPLEX일 때만 생각 켜기
