# 06. 이미지 이해하기

사진·스크린샷을 설명하고, 이미지 속 글자를 읽고(OCR), 도형 위치를 JSON으로 뽑습니다. CLI 한 줄로도, OpenAI SDK의 `image_url`로도 됩니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 이미지 설명, OCR, 위치 정보 JSON |
| 모델 | OCR·좌표: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (2.9 GB) / 한국어 설명: `mlx-community/Qwen3-VL-2B-Instruct-4bit` (1.7 GB) |
| 메모리 | 8 GB 이상 |
| 배우는 것 | VLM 입력 방식, 모델 크기와 작업 궁합 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18. 이미지는 저장소의 `tests/fixtures/test_image_shapes.png`(도형 3개)와 `tests/fixtures/got_ocr_page.png`(글자 두 줄)를 썼습니다.

## 1. CLI로 바로

```bash
mlxcel generate -m mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --image tests/fixtures/test_image_shapes.png \
  -p "List each shape in this image with its color." -n 100 --profile
```

```text
-   A red square is located in the upper left portion of the image.
-   A blue circle is positioned in the upper right.
-   A green triangle is at the bottom of the image, pointing upward.
...
[Profile Results]
  Prompt tokens:    216
  Prefill:          327.69 ms (659.17 tok/s)
  Decode:           528.82 ms (151.28 tok/s)
```

`--image`는 여러 번 줄 수 있습니다. `mlxcel run` 대화 중에는 `/image <경로>`로 붙입니다.

## 2. 서버 + OpenAI SDK

```bash
mlxcel-server -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit --alias qwen25-vl --port 8080
```

```python
# /// script
# dependencies = ["openai>=1.60"]
# ///
import base64
import sys
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")


def image_part(path):
    # Send local files as a data URL; file:// and private http URLs are refused by default.
    data = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def ask(path, question):
    r = client.chat.completions.create(
        model="qwen25-vl", max_tokens=200,
        messages=[{"role": "user", "content": [image_part(path), {"type": "text", "text": question}]}])
    return r.choices[0].message.content.strip()


print("[OCR]", ask(sys.argv[1], "Read all the text in this image exactly."))
print("[JSON]", ask(sys.argv[2], 'Return JSON: [{"shape": ..., "color": ..., "position": ...}] only.'))
print("[KO]", ask(sys.argv[2], "이 그림을 한 문장으로 설명해 줘."))
```

```bash
uv run vision.py tests/fixtures/got_ocr_page.png tests/fixtures/test_image_shapes.png
```

## 실행 결과: 모델에 따라 잘하는 일이 다릅니다

| 작업 | Qwen2.5-VL-3B | Qwen3-VL-2B |
|---|---|---|
| OCR | ✅ `GOT OCR two point zero` / `renders this page` | ❌ `Got 002` / `this` 뒤로 `100`·`111…` 반복 |
| 도형 JSON | ✅ 도형·색·좌표 `[97, 85, 259, 259]` | ✅ 도형·색·위치 (`top_left` 등) |
| 한국어 설명 | △ "하단에서 위로 갈수록 작은색감의 대각형…" (어색함) | ✅ "빨간색 정사각형, 파란색 원, 녹색 삼각형이 … 배치된 세 개의 기하학적 형태입니다." |

Qwen3-VL-2B의 OCR 실패는 mlxcel 문제가 아니라 모델 한계입니다. 같은 이미지·같은 모델을 레퍼런스 런타임 mlx-vlm 0.6.17로 돌려도 `Got 002 / this / 1111…`로 똑같이 실패했고, Qwen2.5-VL-3B는 mlx-vlm과 mlxcel 서버 모두 두 줄을 정확히 읽었습니다.

**추천**: 글자 읽기·좌표가 중요하면 Qwen2.5-VL-3B, 짧은 한국어 설명이면 Qwen3-VL-2B.

## 이미지 보내는 방법

| 방식 | 기본 동작 |
|---|---|
| `data:image/png;base64,...` | ✅ 항상 가능 (위 코드) |
| `file://...` | 서버를 `--media-path <폴더>`로 띄워야 하고, 그 폴더 안만 허용 |
| `http(s)://...` | 공개 주소만 허용. 사설·루프백 주소는 `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`이 필요 |

## 주의

- CLI는 프롬프트를 먼저 출력한 뒤 **줄바꿈 없이** 답을 이어 붙입니다(`...exactly.GOT OCR ...`). 출력을 grep으로 거를 때 첫 줄이 같이 지워지지 않게 하세요.
- 한 요청당 이미지는 기본 16장까지입니다.

## 응용

- 스크린샷 → 버그 리포트 초안
- 영수증 사진 → [04번](04-structured-extraction.md)의 JSON 스키마로 바로 추출 (VLM 요청에도 `response_format`을 함께 쓸 수 있는지는 이 검증에서 확인하지 않았습니다)
