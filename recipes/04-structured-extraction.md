# 04. 텍스트에서 깨지지 않는 JSON 뽑기

영수증, 이메일, 로그처럼 형식 없는 텍스트에서 정해진 JSON을 뽑아냅니다. "JSON으로 답해 줘"라고 부탁하는 대신 **JSON 스키마를 서버에 넘기면**, 서버가 스키마를 깨는 토큰을 매 스텝 지워서 결과가 항상 파싱됩니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 영수증 텍스트 → Pydantic 객체 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | `response_format: json_schema`, 구조화 출력의 한계 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 서버

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b --port 8080
```

## 2. 코드

```python
# /// script
# dependencies = ["openai>=1.60", "pydantic>=2"]
# ///
import json
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")


# 1. Describe the shape you want as a Pydantic model.
class Item(BaseModel):
    name: str
    quantity: int
    unit_price: int


class Receipt(BaseModel):
    store: str
    date: str
    items: list[Item]
    total: int


TEXT = """
[영수증] 행복마트 강남점  2026-10-18 14:32
아메리카노 2개 x 4,500
크루아상 1개 x 3,800
생수 3개 x 1,000
합계 15,800원  카드결제
"""

# 2. Pass its JSON schema as response_format. The server masks every token
#    that would break the schema, so the reply always parses.
r = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": f"Extract the receipt as JSON.\n{TEXT}"}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "receipt", "schema": Receipt.model_json_schema(), "strict": True}},
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
receipt = Receipt.model_validate_json(r.choices[0].message.content)
print(json.dumps(receipt.model_dump(), ensure_ascii=False, indent=2))

# 3. Check the numbers yourself: the schema guarantees shape, not arithmetic.
computed = sum(i.quantity * i.unit_price for i in receipt.items)
print(f"items sum {computed:,} vs total {receipt.total:,}:",
      "OK" if computed == receipt.total else "MISMATCH")
```

## 실행 결과

```text
{
  "store": "행복마트 강남점",
  "date": "2026-10-18",
  "items": [
    {"name": "아메리카노", "quantity": 2, "unit_price": 4500},
    {"name": "크루아상", "quantity": 1, "unit_price": 3800},
    {"name": "생수", "quantity": 3, "unit_price": 1000}
  ],
  "total": 15800
}
items sum 15,800 vs total 15,800: OK
```

"4,500"처럼 쉼표가 들어간 금액도 정수 `4500`으로 바뀌었습니다. 스키마가 `int`를 요구하므로 모델은 숫자 외의 토큰을 고를 수 없습니다.

## `json_object`는 안 됩니다

OpenAI의 `{"type": "json_object"}`는 지원하지 않고 400을 돌려줍니다. 반드시 스키마를 주세요.

```text
{"error":{"message":"invalid response_format: response_format type \"json_object\" is not supported;
 supply type=\"json_schema\" with a schema","type":"invalid_request_error","code":null}}
```

## 배운 것

- **형식은 보장, 내용은 아님.** 스키마는 JSON 모양을 보장하지만 숫자가 맞는지는 보장하지 않습니다. 그래서 3단계처럼 합계를 직접 검산합니다.
- 서버 전체에 같은 스키마를 걸고 싶으면 `--json-schema-file schema.json`, llama.cpp 문법을 쓰려면 `--grammar-file`을 씁니다.
- 원리: 매 스텝 허용되지 않는 토큰의 logit을 −∞로 만든 뒤 샘플링합니다(llguidance, `src/server/structured.rs`).

## 응용

- 고객 문의 메일 → `{카테고리, 긴급도, 요약}` 분류
- 로그 한 줄 → `{시간, 레벨, 서비스, 메시지}` 파싱
- 03번 에이전트의 최종 답을 JSON으로 받아 다른 시스템에 넘기기
