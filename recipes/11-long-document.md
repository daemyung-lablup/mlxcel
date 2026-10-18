# 11. 긴 문서 요약: 메모리부터 계산하기

긴 문서를 넣기 전에 **필요한 메모리를 먼저 계산**하고, 실제로 1만 7천 토큰짜리 문서를 요약해 봅니다. 문맥이 길어질수록 커지는 KV 캐시를 INT8로 줄이는 방법도 비교합니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 메모리 사전 점검 + 17K 토큰 문서 요약 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 16 GB 이상 (32K 문맥 기준) |
| 배우는 것 | `mlxcel inspect`, KV 캐시 크기, `--kv-cache-mode` |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 올리기 전에 메모리 계산

```bash
mlxcel inspect -m mlx-community/Qwen3-4B-4bit --max-tokens 4096
```

```text
  Context length:  4096 tokens (batch = 1)
  KV dtype:        fp16
  Weights:         2.11 GiB (safetensors header)
  KV cache:        576.0 MiB (standard attention (36 layers, full context))
                   (147456 bytes per token at steady state, same dtype)
  Activation:      96.3 MiB
  Allocator ovhd:  546.8 MiB (factor 1.20x on weights+kv)
  Total estimate:  3.30 GiB
  Available:       45.60 GiB
  FITS: 42.30 GiB of headroom
```

문맥 길이, KV 형식, 동시 요청 수를 바꿔 가며 비교했습니다.

| 조건 | KV 캐시 | 합계 | 판정 (48 GB Mac) |
|---|---|---|---|
| 4K, fp16 | 576 MiB | 3.30 GiB | ✅ |
| 32K, fp16 | 4.50 GiB | 8.02 GiB | ✅ |
| 32K, **int8** (`--kv-cache-mode int8`) | **2.25 GiB** | 5.32 GiB | ✅ |
| 32K × 동시 8개 (`--batch 8`) | 36.00 GiB | 46.48 GiB | ❌ 902 MiB 초과 |

- 토큰 하나당 KV는 144 KiB입니다. **약 1만 5천 토큰이면 KV 캐시가 모델 가중치(2.11 GiB)보다 커집니다.**
- 스크립트나 스케줄러에서 쓰려면 `--json`을 붙입니다.

  ```bash
  mlxcel inspect -m mlx-community/Qwen3-4B-4bit --max-tokens 32768 --json
  # {"weights_bytes": 2262920192, "kv_bytes_total": 4831838208, "total_bytes": 8614677247, "fits": true, ...}
  ```

## 2. 서버와 요약 코드

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b --port 8080
# 메모리를 아끼려면: --kv-cache-mode int8
```

```python
# /// script
# dependencies = ["openai>=1.60", "httpx"]
# ///
import sys
import time
import httpx
from openai import OpenAI

BASE = "http://localhost:8080"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")
text = open(sys.argv[1], encoding="utf-8").read()

# 1. Count tokens with the server's own tokenizer before sending.
n_tokens = len(httpx.post(f"{BASE}/tokenize", json={"content": text}).json()["tokens"])
print(f"document: {len(text):,} chars = {n_tokens:,} tokens")

# 2. Summarize. Long prompts are prefilled in chunks before the first token.
t = time.perf_counter()
r = client.chat.completions.create(
    model="qwen3-4b", max_tokens=400,
    messages=[{"role": "user", "content":
               f"Summarize this document in 5 bullet points.\n\n{text}"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(f"total {time.perf_counter() - t:.1f}s, prompt {r.usage.prompt_tokens:,} tokens")
print(r.choices[0].message.content)
```

```bash
uv run summarize.py path/to/mlxcel/docs/CONTINUOUS_BATCHING.md
```

## 실행 결과

```text
document: 64,710 chars = 17,037 tokens
total 39.9s, prompt 17,060 tokens
- **Continuous Batching and Disaggregated Serving**: `mlxcel-server` manages continuous batching
  with a scheduler that interleaves prefill and decode, using a paged KV cache ...
- **Batching Scheduler and KV Cache**: The scheduler admits requests, interleaves prefill and
  decode, and uses a paged KV cache for efficient memory management ...
- **Serving Throughput and Token Exactness**: ...
- **Disaggregated Serving Architecture**: Prefill and decode operations are split into separate
  processes (or machines), with a router handling client requests ...
- **Key Features and Trade-offs**: ...
```

17K 토큰 문서 하나에 약 40초가 걸렸고, 대부분은 프롬프트를 처리하는 프리필 시간입니다.

## INT8 KV는 메모리용이지 속도용이 아닙니다

KV 캐시를 INT8로 저장하면 메모리는 절반이 되지만, 매 스텝 다시 풀어야 해서 디코드는 느려집니다. mlxcel 문서의 측정(Qwen3-8B, M1 Ultra)으로는 FP16 대비 4K 문맥에서 0.61배, 16K에서 0.41배였습니다(`docs/turbo-kv-cache.md`). **메모리가 모자랄 때만** 켜세요. 이 레시피에서는 INT8 서버로 직접 요약 시간을 재지는 않았습니다.

## 주의

- 서버의 최대 문맥은 기본으로 모델 설정에서 정해집니다. 직접 정하려면 `-c/--ctx-size`를 씁니다.
- 동시 요청(`--parallel`)을 늘리면 KV 캐시도 그만큼 곱해집니다. 위 표의 `--batch 8`처럼 미리 계산해 보세요.

## 응용

- 이 문서를 넘는 길이는 장 단위로 나눠 요약한 뒤 다시 요약하기(map-reduce)
- 같은 긴 문서에 여러 질문을 던진다면, 앞부분이 같아서 프리픽스 캐시가 두 번째부터 프리필을 건너뜁니다.
