# 08. 코드 수정을 빠르게: prompt lookup

변수 이름 바꾸기, JSON 필드 고치기처럼 **입력을 거의 그대로 다시 쓰는 작업**은 prompt lookup으로 빨라집니다. 드래프트 모델 없이, 프롬프트 안에 이미 있는 n-gram을 복사해 추측하고 큰 모델이 한 번에 검증합니다. 반대로 자유 작문에서는 이득이 없거나 손해입니다. 두 경우를 모두 재 봅니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 같은 요청을 prompt lookup 켜고 끄고 비교 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | 스펙큘레이티브 디코딩, 채택률, 적응형 정책 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. CLI로 비교

코드 수정 요청: 40줄 정도의 Python 코드를 주고 변수 이름을 바꾸게 합니다(`/no_think`로 thinking 끔).

```bash
SRC=$(cat my_script.py)
P="Rename the variable prompt_ids to token_ids everywhere in this Python code. Output only the full updated code. /no_think

$SRC"

mlxcel generate -m mlx-community/Qwen3-4B-4bit -p "$P" -n 500 --profile
mlxcel generate -m mlx-community/Qwen3-4B-4bit -p "$P" -n 500 --profile --prompt-lookup
```

자유 작문 요청:

```bash
S="Write a short story about a cat who learns to fly. /no_think"
mlxcel generate -m mlx-community/Qwen3-4B-4bit -p "$S" -n 500 --profile
mlxcel generate -m mlx-community/Qwen3-4B-4bit -p "$S" -n 500 --profile --prompt-lookup
```

## 실행 결과

```text
== code plain
  Decode:           5890.52 ms (84.88 tok/s)
== code --prompt-lookup
[Prompt lookup] rounds=81 drafted_rounds=71 paused_rounds=0 proposed=497 accepted=418
                acceptance_rate=0.8410 tokens_per_forward=6.1605
  Decode:           2470.66 ms (201.97 tok/s)

== story plain
  Decode:           5641.37 ms (88.63 tok/s)
== story --prompt-lookup
[Prompt lookup] rounds=406 drafted_rounds=16 paused_rounds=32 proposed=55 accepted=3
                acceptance_rate=0.0545 tokens_per_forward=1.0049
  Decode:           5061.51 ms (80.61 tok/s)
```

| 작업 | 끔 | 켬 | 배수 | 채택률 | forward당 토큰 |
|---|---|---|---|---|---|
| 코드 수정 | 84.9 tok/s | 202.0 tok/s | **2.38×** | 84% | 6.16 |
| 자유 작문 | 88.6 tok/s | 80.6 tok/s | **0.91×** | 5% | 1.00 |

자유 작문에서는 추측이 거의 맞지 않자 적응형 정책이 32라운드를 쉬어 갔지만(`paused_rounds=32`), 그래도 조금 느려졌습니다.

## 2. 서버에서 켜기

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b \
  --draft-kind prompt-lookup --port 8080
```

같은 코드 수정 요청을 `/v1/chat/completions`로 보내면, 응답의 `timings`에 드래프트 통계가 붙습니다.

```json
{
 "prompt_n": 538,
 "predicted_n": 500,
 "predicted_per_second": 166.0,
 "draft_n": 513,
 "draft_n_accepted": 420,
 "draft_rounds": 74,
 "draft_kind": "prompt-lookup"
}
```

평균 채택 길이는 `(draft_n_accepted + draft_rounds) / draft_rounds` = (420 + 74) / 74 ≈ 6.7토큰입니다.

> 💡 플래그가 다릅니다. **CLI는 `--prompt-lookup`, 서버는 `--draft-kind prompt-lookup`** 입니다. `mlxcel generate --draft-kind prompt-lookup`은 `--prompt-lookup`을 쓰라는 오류를 냅니다.

## 주의

- 이미지 요청, 구조화 출력(`response_format`), logprobs 요청, Mamba 같은 하이브리드 모델에서는 자동으로 일반 디코딩으로 돌아갑니다.
- 작문처럼 다음 토큰을 예측하기 어려운 작업에 항상 켜 두면 손해일 수 있습니다. 편집·변환 위주의 서버에만 켜세요.

## 배운 것

- 디코드는 가중치를 읽는 시간이 지배적이라, 토큰 여러 개를 forward 한 번에 검증해도 비용이 거의 같습니다. 그래서 추측이 맞으면 그만큼 빨라집니다.
- 이득은 **출력이 입력을 얼마나 되풀이하느냐**에 달려 있습니다.

## 응용

- 코드 리뷰 반영, 번역문 교정, 설정 파일 일괄 수정처럼 "거의 같은 텍스트를 다시 쓰는" 작업
- 드래프트 모델을 쓰는 방식(MTP, DFlash)은 Gemma 4, Qwen 3.5 같은 지원 모델에서 `--draft-model`로 씁니다.
