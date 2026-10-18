# 12. LoRA 어댑터 만들고 갈아 끼우기

mlx-lm으로 작은 LoRA 어댑터를 직접 학습시키고, mlxcel 서버에 올린 뒤 **서버를 재시작하지 않고** 어댑터 세기를 바꿔 봅니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | "해적 말투" 어댑터 학습 → 서빙 → 세기 0 / 0.5 / 1 비교 |
| 모델 | `mlx-community/Qwen3-0.6B-4bit` |
| 메모리 | 8 GB 이상 (학습 피크 0.93 GB) |
| 배우는 것 | LoRA 학습, `--lora`, `POST /lora-adapters`, 과적합 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), mlx-lm 0.31.3, Apple M4 Pro 48 GB, 2026-10-18

## 1. 학습 데이터 만들기

한 가지 습관(해적 말투)만 가르치는 80개짜리 대화 데이터입니다.

```python
# make_data.py: build a tiny chat dataset that teaches one habit: answer like a pirate.
import json, random

random.seed(0)
topics = ["the weather", "Python lists", "coffee", "the moon", "trains", "cats", "the ocean",
          "music", "a good breakfast", "learning to code", "rain", "books", "the stars",
          "bicycles", "tea", "mountains", "the internet", "gardening", "chess", "bread"]
asks = ["Tell me about {}.", "What do you think of {}?", "Give me one fact about {}.",
        "Describe {} in one sentence."]
rows = []
for t in topics:
    for a in asks:
        q = a.format(t)
        ans = (f"Arr, matey! {t.capitalize()} be a fine thing to ponder on the high seas. "
               f"Any sailor worth his salt knows a tale or two about {t}, yo ho!")
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": ans}]})
random.shuffle(rows)
with open("train.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows[:70])
with open("valid.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows[70:])
print(len(rows), "examples")
```

```bash
python3 make_data.py && mkdir -p data && mv train.jsonl valid.jsonl data/
```

## 2. 학습 (약 10초)

```bash
mlxcel download mlx-community/Qwen3-0.6B-4bit --force

uvx --from mlx-lm==0.31.3 mlx_lm.lora \
  --model ~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit \
  --train --data data --iters 100 --batch-size 4 --num-layers 8 \
  --adapter-path pirate-adapter
```

```text
Iter 1: Val loss 5.552
Iter 50: Train loss 0.180, It/sec 10.804, Tokens/sec 2366.131, Peak mem 0.928 GB
Iter 100: Val loss 0.133
Saved final weights to pirate-adapter/adapters.safetensors.
```

결과 폴더에는 `adapter_config.json`과 `adapters.safetensors`가 생기고, mlxcel은 이 형식을 그대로 읽습니다.

## 3. 서빙

```bash
mlxcel-server -m mlx-community/Qwen3-0.6B-4bit --alias qwen3-0.6b \
  --lora ./pirate-adapter --port 8080
```

```text
Staged runtime LoRA adapter .../pirate-adapter for 56 layers (rank scale 20.000, user scale 1.00)
```

## 4. 세기를 바꿔 가며 묻기

```python
# /// script
# dependencies = ["openai>=1.60", "httpx"]
# ///
import httpx
from openai import OpenAI

BASE = "http://localhost:8080"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")

print("adapters:", httpx.get(f"{BASE}/lora-adapters").json())


def ask(question):
    r = client.chat.completions.create(
        model="qwen3-0.6b", max_tokens=60, temperature=0,
        messages=[{"role": "user", "content": question}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content.strip().replace("\n", " ")


# Change the adapter's strength live: no restart, no reload.
for scale in [1.0, 0.0, 0.5]:
    httpx.post(f"{BASE}/lora-adapters", json=[{"id": 0, "scale": scale}]).raise_for_status()
    print(f"\nscale {scale}:")
    for q in ["Tell me about volcanoes.", "What is a good name for a dog?"]:
        print(f"  {q} -> {ask(q)[:110]}")
```

## 실행 결과

```text
adapters: [{'id': 0, 'path': '.../pirate-adapter', 'scale': 1.0, 'task_name': '', 'prompt_prefix': ''}]

scale 1.0:
  Tell me about volcanoes. -> Arr, matey! Volcans be a fine thing to ponder on the high seas. Any sailor worth his salt knows a tale or two
  What is a good name for a dog? -> Arr, matey! A good question to ponder on the high seas. Any sailor worth his salt knows a tale or two about a

scale 0.0:
  Tell me about volcanoes. -> Volcanoes are natural formations on Earth, formed by the movement of tectonic plates and the eruption of magma
  What is a good name for a dog? -> A good name for a dog is not a name, but a name that reflects the dog's personality and characteristics. Howev

scale 0.5:
  Tell me about volcanoes. -> Arr, volcanoes! They're such a fine thing to ponder on. Any sailor worth his salt knows a tale or two about vo
  What is a good name for a dog? -> Arr, matey! A good name for a dog... Well, a dog is a dog. Any good name for a dog?
```

- **1.0**: 해적 말투. 학습에 없던 질문(화산, 개 이름)에도 적용됩니다.
- **0.0**: 원래 모델과 같습니다. 어댑터를 끈 것과 같습니다.
- **0.5**: 말투는 남고 내용이 조금 살아납니다.

## 배운 것: 데이터가 작으면 틀을 외운다

80개 예시가 모두 같은 문장 틀이라, 어댑터는 말투가 아니라 **문장 자체를 외웠습니다**(검증 손실 0.13). 그래서 1.0에서는 무엇을 물어도 "fine thing to ponder on the high seas"가 나옵니다. 실제로 쓸 어댑터는 다양한 답변 예시로 학습하고, 세기를 0.5처럼 낮춰 균형을 잡을 수 있습니다.

## 주의

- 4비트 같은 양자화 모델에는 위처럼 융합하지 않는 `--lora`를 씁니다. `--lora-fuse`는 디코드 비용이 없지만 세기를 바꿀 수 없습니다.
- 어댑터를 여러 개 올리려면 `--lora a,b`로 주고 `POST /lora-adapters`에 `[{"id": 0, "scale": ...}, {"id": 1, "scale": ...}]`를 보냅니다. 목록에 없는 어댑터는 0이 됩니다.
- 요청마다 다른 어댑터를 쓰는 `lora` 필드는 네이티브 `/completion`에만 있고 `/v1/chat/completions`에는 없습니다. 이 검증에서는 서버 전체 세기만 바꿔 봤습니다.
- DoRA와 MoE 전문가 어댑터는 지원하지 않습니다.

## 응용

- 사내 문체, 고객 응대 말투, 특정 출력 형식을 어댑터로 학습해 두고 상황에 따라 세기 조절
- 같은 베이스 모델 하나에 어댑터 여러 개를 올려 메모리를 아끼기
