# 02. 기존 OpenAI·Anthropic 앱을 로컬 모델로

이미 OpenAI SDK나 Anthropic SDK로 짠 코드가 있다면, **`base_url` 한 줄만 바꿔서** 내 Mac의 모델로 돌릴 수 있습니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 같은 코드로 OpenAI 형식, Anthropic 형식, 스트리밍 호출 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | API 호환, `--alias`, thinking 끄기 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 서버 띄우기

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b --port 8080
```

> 💡 `--alias`를 꼭 주세요. HuggingFace 캐시에서 불러온 모델은 이름이 스냅샷 해시(`4dcb3d10...`)로 나와서, `model` 필드에 무엇을 넣어야 할지 알기 어렵습니다. `GET /v1/models`로 실제 이름을 확인할 수 있습니다.

## 2. 코드

```python
# /// script
# dependencies = ["openai>=1.60", "anthropic>=0.40"]
# ///
from openai import OpenAI
from anthropic import Anthropic

# OpenAI SDK: only base_url changes. Any api_key works when the server has none.
openai = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
r = openai.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Name three planets. One line."}],
    max_tokens=64,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("OpenAI   :", r.choices[0].message.content.strip())

# Anthropic SDK: base_url is the server root; the SDK adds /v1/messages.
anthropic = Anthropic(base_url="http://localhost:8080", api_key="local")
m = anthropic.messages.create(
    model="qwen3-4b",
    max_tokens=64,
    messages=[{"role": "user", "content": "Name three planets. One line. /no_think"}],
)
print("Anthropic:", "".join(b.text for b in m.content if b.type == "text").strip())

# Streaming works the same way.
print("Stream   : ", end="")
for chunk in openai.chat.completions.create(
    model="qwen3-4b", stream=True, max_tokens=32,
    messages=[{"role": "user", "content": "Count to five."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
):
    print(chunk.choices[0].delta.content or "", end="", flush=True) if chunk.choices else None
print()
```

```bash
uv run sdk.py
```

## 실행 결과

```text
OpenAI   : Earth, Mars, Jupiter.
Anthropic: Earth, Mars, Jupiter.
Stream   : 1. One
2. Two
3. Three
4. Four
5. Five
```

## 바꿔야 하는 것은 세 가지뿐

| | OpenAI SDK | Anthropic SDK |
|---|---|---|
| `base_url` | `http://localhost:8080/v1` | `http://localhost:8080` (SDK가 `/v1/messages`를 붙임) |
| `api_key` | 서버에 키가 없으면 아무 값 | 같음 (`x-api-key` 헤더도 받음) |
| `model` | `--alias`로 준 이름 | 같음 |

## 주의

- Qwen3는 기본으로 생각(thinking)부터 합니다. 짧은 답이 필요하면 OpenAI 쪽은 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, Anthropic 쪽은 메시지 끝에 `/no_think`를 붙였습니다 ([09번](09-thinking.md)).
- 서버를 `--api-key`나 `--api-key-file`로 띄우면 그 키를 `api_key`에 넣으세요. `Authorization: Bearer`와 `x-api-key` 둘 다 받습니다.

## 응용

- LangChain, LlamaIndex처럼 OpenAI 호환 엔드포인트를 받는 프레임워크도 같은 방식으로 붙습니다.
- 도구 호출 → [03번](03-tool-agent.md), JSON 출력 → [04번](04-structured-extraction.md)
