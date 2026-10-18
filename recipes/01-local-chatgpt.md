# 01. 5분 만에 로컬 ChatGPT

브라우저에서 쓰는 채팅 UI를 내 Mac에 띄웁니다. 모델 목록, 로드·언로드, 다운로드, 채팅이 `mlxcel-server`에 들어 있어서 따로 설치할 것이 없습니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | `http://127.0.0.1:8080/webui/` 에서 동작하는 채팅 UI |
| 모델 | `mlx-community/Qwen3-0.6B-4bit` (335 MB) 또는 `mlx-community/Qwen3-4B-4bit` (2.1 GB) |
| 메모리 | 8 GB 이상 |
| 배우는 것 | 라우터 모드, API 키, 모델 저장소 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, macOS, 2026-10-18

## 1. 모델 받기

```bash
mlxcel download mlx-community/Qwen3-0.6B-4bit
mlxcel download mlx-community/Qwen3-4B-4bit
```

받은 모델은 `~/.cache/mlxcel/models/<owner>/<name>` 에 저장되고, WebUI 라이브러리에 자동으로 나타납니다.

> ⚠️ 이미 HuggingFace 캐시(`~/.cache/huggingface/hub`)에 같은 모델이 있으면 `mlxcel download`는 "reusing without re-download"라고 하고 끝납니다. 이 상태에서는 WebUI·라우터 목록에 **나타나지 않습니다**. 이때는 `--force`를 붙여 mlxcel 저장소에 실제 파일로 받으세요.
>
> ```bash
> mlxcel download mlx-community/Qwen3-0.6B-4bit --force
> ```

## 2. 서버 띄우기

터미널에서 직접 실행하면 1회용 세션 키가 출력되고, 그 키를 브라우저에 붙여 넣으면 됩니다.

```bash
mkdir -p ~/models/mlx
mlxcel-server --webui --models-dir ~/models/mlx --port 8080
# mlxcel WebUI session key (shown once): mlxcel_webui_...
```

서비스로 띄우거나 키를 고정하고 싶으면 키 파일을 씁니다. 이 키는 일반 API 호출에도 그대로 쓰입니다.

```bash
mkdir -p ~/.config/mlxcel ~/models/mlx
(umask 077 && openssl rand -hex 32 > ~/.config/mlxcel/webui.key)
mlxcel-server --webui --api-key-file ~/.config/mlxcel/webui.key \
  --models-dir ~/models/mlx --port 8080
```

`--models-dir`의 바로 아래 폴더 중 `config.json`이 있는 것과 mlxcel 저장소의 모델이 함께 목록에 나옵니다. `-m` 없이 `--webui`로 띄우면 라우터 모드가 되고, 모델은 UI에서 **Load**를 눌러야 메모리에 올라갑니다.

## 3. 브라우저로 열기

```
http://127.0.0.1:8080/webui/
```

> ⚠️ `localhost:8080`으로 열면 **403 "Host header is not allowed"** 가 나옵니다. 서버 로그에 찍힌 주소(`127.0.0.1`) 그대로 여세요.
>
> 이건 **버그**입니다. 시작할 때는 `127.0.0.1`·`localhost`·`[::1]`을 모두 허용 목록에 넣는데, 포트에 바인딩한 뒤 `update_exact_origin`이 목록을 바인딩 주소 하나로 덮어씁니다(`src/server/webui/security/policy.rs`, `src/server/startup.rs:2326`). `--webui`를 켠 서버에서는 `/health`, `/v1/*`도 같은 이유로 `localhost`에서 403이 납니다.

## 실행 결과 (키 파일 방식)

```text
$ curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/health
403                                    ← localhost는 거부
$ curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/webui/
200 (text/html)
$ curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models
401                                    ← 키 없으면 거부
$ curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/v1/models
  mlx-community/Qwen3-0.6B-4bit unloaded
  mlx-community/Qwen3-4B-4bit unloaded
  ... (저장소의 모델 9개)
$ curl -s -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/models/load \
    -d '{"model":"mlx-community/Qwen3-0.6B-4bit"}'
{"success":true}
chat: Hello! How can I assist you today?
```

UI에서 Load 버튼을 누르는 것과 같은 동작을 `POST /models/load`로 확인했습니다. 대화형 터미널의 세션 키 출력과 화면 자체는 이 검증에서 확인하지 않았습니다.

## 배운 것

- `--webui`만 주고 `-m`을 빼면 **라우터 모드**: 여러 모델을 목록으로 두고 필요할 때 올립니다 ([10번 레시피](10-router.md)).
- WebUI와 API는 같은 키를 씁니다. 키 파일 하나로 브라우저와 스크립트를 함께 쓸 수 있습니다.

## 응용

- 같은 서버에 OpenAI SDK를 붙여 스크립트에서도 쓰기 → [02번](02-openai-anthropic-sdk.md)
- 원격 접속은 TLS(`--ssl-cert-file`, `--ssl-key-file`)나 리버스 프록시를 거치세요. 자세한 내용은 `docs/webui/README.md`.
