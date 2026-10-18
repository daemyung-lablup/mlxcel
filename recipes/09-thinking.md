# 09. 추론(thinking) 모델 다루기

Qwen3 같은 추론 모델은 답하기 전에 `<think>` 블록에서 생각부터 합니다. 생각을 켜고 끄고, 길이를 제한하고, 생각과 답을 나눠 받는 방법을 비교합니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 같은 질문을 세 가지 설정으로 보내 시간·토큰 비교 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | `enable_thinking`, `thinking_budget_tokens`, `reasoning_content` |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 서버

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b --port 8080
```

## 2. 코드

```python
# /// script
# dependencies = ["openai>=1.60"]
# ///
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
QUESTION = "Is 391 a prime number? Answer in one sentence."


def run(label, **extra):
    t = time.perf_counter()
    r = client.chat.completions.create(
        model="qwen3-4b", max_tokens=2048,
        messages=[{"role": "user", "content": QUESTION}],
        temperature=0.6, top_p=0.95, extra_body={"top_k": 20, **extra},
    )
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    print(f"[{label}] {time.perf_counter() - t:.1f}s, "
          f"{r.usage.completion_tokens} tokens, reasoning {len(reasoning)} chars")
    print(f"  answer: {msg.content.strip()}")


# 1. Thinking on (Qwen3 default): reasoning arrives in reasoning_content.
run("thinking on")
# 2. Thinking off for this request only.
run("thinking off", chat_template_kwargs={"enable_thinking": False})
# 3. Thinking on, but capped: the server closes </think> after 64 tokens.
run("budget 64", thinking_budget_tokens=64)
```

## 실행 결과

```text
[thinking on] 8.4s, 623 tokens, reasoning 1458 chars
  answer: No, 391 is not a prime number because it can be factored into 17 and 23 (17 × 23 = 391).
[thinking off] 0.4s, 26 tokens, reasoning 0 chars
  answer: No, 391 is not a prime number because it can be factored into 17 × 23.
[budget 64] 1.2s, 93 tokens, reasoning 230 chars
  answer: No, 391 is not a prime number because it can be factored into 17 × 23.
```

| 설정 | 시간 | 출력 토큰 | 생각 |
|---|---|---|---|
| 켬 (기본) | 8.4초 | 623 | 1,458자 |
| 끔 | 0.4초 | 26 | 없음 |
| 예산 64토큰 | 1.2초 | 93 | 230자 |

이 쉬운 질문에서는 세 답이 모두 맞았지만, 생각을 켜면 **20배 넘게** 오래 걸렸습니다. 어려운 문제일수록 생각의 이득이 커지므로, 작업에 맞게 고르세요.

## 설정 방법 정리

아래는 모두 직접 실행해 확인했습니다(질문: "Is 391 prime? One sentence.").

| 하고 싶은 것 | 방법 | 확인 결과 |
|---|---|---|
| 요청마다 생각 끄기 | `chat_template_kwargs: {"enable_thinking": false}` | 26토큰, 생각 없음 |
| 〃 (다른 표기) | `reasoning_effort: "none"` | 22토큰, 생각 없음 |
| 서버 기본값으로 끄기 | `--chat-template-kwargs '{"enable_thinking":false}'` | 22토큰, 생각 없음 |
| 요청마다 생각 길이 제한 | `thinking_budget_tokens: 64` | 93토큰, 생각 230자, 깔끔한 답 |
| 서버 기본값으로 제한 | `--reasoning-budget 32` (0이면 바로 닫음, -1이면 무제한) | ⚠️ 아래 주의 참고 |
| 생각을 답에 그대로 남기기 | `--reasoning-format none` | `content`가 `<think>\n...`로 시작 |
| 스트리밍 | 기본 | 생각은 `delta.reasoning_content`로 옴 (194개 청크) |

## 주의

- **예산을 너무 작게 잡으면 생각이 답으로 넘쳐 나옵니다.** `--reasoning-budget 32`에서 생각 블록은 닫혔지만, 모델이 답변 칸에서 "First, check if 391 is divisible by small primes..."처럼 생각을 계속 이어 써서 432토큰이 나왔습니다. 64처럼 조금 넉넉하게 잡거나, 아예 끄는 편이 낫습니다.
- **생각을 끄면 답이 거칠어질 수 있습니다.** `reasoning_effort: "none"`으로 받은 답은 "Yes, 391 is not prime; it is divisible by 17 and 23."처럼 앞뒤가 맞지 않았습니다(결론은 맞음).
- 생각도 출력 토큰입니다. `max_tokens`를 작게 주면 생각 도중에 끊겨 답이 비어 있을 수 있습니다.
- Qwen3는 생각 모드에서 greedy(temperature 0) 대신 위 코드처럼 temp 0.6, top_p 0.95, top_k 20을 권장합니다. greedy로 두면 긴 생각에서 같은 말을 반복하기 쉽습니다.
- Anthropic SDK에서는 `chat_template_kwargs`를 넘기기 어려우니 메시지 끝에 `/no_think`를 붙이는 방법도 됩니다([02번](02-openai-anthropic-sdk.md)).

## 응용

- 라우터([10번](10-router.md))와 묶어서, 쉬운 질문은 생각 끄고 어려운 질문만 생각 켜기
- 사용자에게는 답만 보여 주고 `reasoning_content`는 로그로 남겨 디버깅에 쓰기
