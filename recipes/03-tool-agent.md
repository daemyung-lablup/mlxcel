# 03. 도구를 쓰는 에이전트

날씨 조회, 계산기, 파일 목록 같은 Python 함수를 모델이 골라 호출하는 작은 비서를 만듭니다. 서버가 모델 출력의 `<tool_call>`을 OpenAI `tool_calls` 형식으로 바꿔 주므로, 클라이언트는 **함수를 실행하고 결과를 돌려주는 루프**만 쓰면 됩니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 도구 3개를 쓰는 에이전트 루프 |
| 모델 | `mlx-community/Qwen3-4B-4bit` |
| 메모리 | 8 GB 이상 |
| 배우는 것 | tool calling, `tool_choice`, enum 스키마, 모델이 암산하는 문제 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 서버

```bash
mlxcel-server -m mlx-community/Qwen3-4B-4bit --alias qwen3-4b --port 8080
```

별도 플래그는 필요 없습니다. tool call 파싱은 기본으로 켜져 있습니다.

## 2. 코드

```python
# /// script
# dependencies = ["openai>=1.60"]
# ///
import json
import pathlib
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
MODEL = "qwen3-4b"


# 1. Real functions the agent may call.
def get_weather(city: str) -> dict:
    fake = {"Seoul": (18, "clear"), "Busan": (21, "cloudy")}
    if city not in fake:
        return {"error": f"unknown city {city!r}; use one of {sorted(fake)}"}
    temp, sky = fake[city]
    return {"city": city, "temperature_c": temp, "sky": sky}


def calculate(expression: str) -> dict:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "only numbers and + - * / ( ) are allowed"}
    return {"result": eval(expression)}  # safe: characters checked above


def list_files(pattern: str) -> dict:
    return {"files": sorted(p.name for p in pathlib.Path(".").glob(pattern))[:20]}


FUNCTIONS = {"get_weather": get_weather, "calculate": calculate, "list_files": list_files}


def tool(name, description, **params):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": params, "required": list(params)}}}


# 2. JSON schemas the model reads. An enum keeps city names valid.
TOOLS = [
    tool("get_weather", "Current weather for a city.",
         city={"type": "string", "enum": ["Seoul", "Busan"]}),
    tool("calculate", "Evaluate an arithmetic expression.",
         expression={"type": "string"}),
    tool("list_files", "List files in the current folder matching a glob.",
         pattern={"type": "string", "description": "e.g. *.py"}),
]


# 3. The agent loop: ask, run requested tools, feed results back, repeat.
def ask(question: str, max_rounds: int = 5) -> str:
    messages = [
        {"role": "system", "content": "Use the calculate tool for any arithmetic. "
                                      "Never compute numbers yourself."},
        {"role": "user", "content": question},
    ]
    for _ in range(max_rounds):
        reply = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message
        if not reply.tool_calls:
            return reply.content
        messages.append(reply)
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = FUNCTIONS[call.function.name](**args)
            print(f"  -> {call.function.name}({args}) = {result}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return "(stopped: too many rounds)"


for q in ["서울과 부산 날씨 알려줘.",
          "1234 곱하기 5678에 99를 더하면?",
          "이 폴더에 있는 파이썬 파일 목록 보여줘."]:
    print(f"Q: {q}")
    print(f"A: {ask(q)}\n")
```

## 실행 결과

```text
Q: 서울과 부산 날씨 알려줘.
  -> get_weather({'city': 'Seoul'}) = {'city': 'Seoul', 'temperature_c': 18, 'sky': 'clear'}
  -> get_weather({'city': 'Busan'}) = {'city': 'Busan', 'temperature_c': 21, 'sky': 'cloudy'}
A: 서울의 현재 날씨는 18°C, 맑은 하늘입니다. 부산의 현재 날씨는 21°C, 흐린 날씨입니다.

Q: 1234 곱하기 5678에 99를 더하면?
  -> calculate({'expression': '1234 * 5678 + 99'}) = {'result': 7006751}
A: 1234 곱하기 5678에 99를 더하면 7006751입니다.

Q: 이 폴더에 있는 파이썬 파일 목록 보여줘.
  -> list_files({'pattern': '*.py'}) = {'files': ['r02_sdk.py', 'r03_agent.py', ...]}
A: 이 폴더에 있는 파이썬 파일은 다음과 같습니다: ...
```

모델은 두 도시를 **한 번에 병렬로** 호출했고, 한국어 질문인데도 enum 덕분에 `"Seoul"`처럼 정확한 값을 넘겼습니다.

## 꼭 알아야 할 함정: 모델은 도구를 안 쓰고 암산할 수 있다

system 프롬프트 없이 같은 질문을 하면, 모델은 계산기를 부르지 않고 **틀린 답**을 냈습니다.

```text
Q: 1234 곱하기 5678에 99를 더하면?
A: ... 1234 * 5678 = 7000172 ... So, the final result is 7000271.   ← 정답은 7006751
```

해결 방법은 두 가지이고, 둘 다 확인했습니다.

1. **system 프롬프트로 지시**: 위 코드처럼 "Use the calculate tool for any arithmetic." → 도구 호출, 정답 7006751
2. **`tool_choice`로 강제**: 반드시 특정 도구를 부르게 합니다.

   ```python
   tool_choice={"type": "function", "function": {"name": "calculate"}}
   # -> finish_reason: tool_calls, arguments: {"expression":"1234 * 5678 + 99"}
   ```

   강제된 `tool_choice`는 문법(grammar)으로 출력 형식까지 제한되므로 형식이 깨지지 않습니다.

## 배운 것

- 모델은 텍스트만 씁니다. 실행은 클라이언트가 합니다. 서버(mlxcel)는 `<tool_call>` 텍스트를 `tool_calls`로 바꿔 줄 뿐 함수를 실행하지 않습니다.
- 인자 값은 모델이 쓴 텍스트이므로 **enum으로 제한하고, 함수 안에서 다시 검증**하세요. enum 없이 했을 때 같은 모델이 `"서울"`을 보낸 적이 있습니다.

## 응용

- 사내 API, DB 조회, 파일 읽기 같은 함수를 늘려 가면 됩니다. 쓰기·삭제 같은 위험한 도구는 실행 전 사용자 확인을 넣으세요.
- 도구 결과를 그대로 JSON으로 받고 싶다면 → [04번](04-structured-extraction.md)
