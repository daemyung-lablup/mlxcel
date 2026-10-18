# 07. 읽어 주기와 받아쓰기

Kokoro로 텍스트를 음성으로 만들고(TTS), Whisper로 음성을 텍스트로 받아씁니다(STT). 이 레시피는 만든 음성을 다시 받아써서 **왕복**으로 확인합니다.

| 항목 | 내용 |
|---|---|
| 만드는 것 | 텍스트 → WAV → 텍스트 |
| 모델 | TTS `mlx-community/Kokoro-82M-bf16` (339 MB), STT `openai/whisper-small` (926 MB) |
| 메모리 | 4 GB 이상 |
| 배우는 것 | `/v1/audio/speech`, `/v1/audio/transcriptions`, 모델 크기와 정확도 |

> 검증 환경: mlxcel 0.7.0 (`main` 74f9ae1d), Apple M4 Pro 48 GB, 2026-10-18

## 1. 모델 받기

```bash
mlxcel download mlx-community/Kokoro-82M-bf16
mlxcel download openai/whisper-small
```

## 2. 서버 두 개

음성 모델은 채팅 모델과 같은 프로세스에 올릴 수 없어서, TTS와 STT를 각각 띄웁니다.

### TTS (Kokoro): 지금은 우회가 필요합니다

> ⚠️ **현재 `main`에서는 문서대로 띄우면 실패합니다.**
>
> ```text
> Error: No tokenizer found in ".../Kokoro-82M-bf16" (tried tokenizer.json, tokenizer.model, ...)
> ```
>
> 서버가 Kokoro 체크포인트인지 확인하기 전에 채팅용 토크나이저부터 읽으려 하기 때문입니다(`src/server/startup.rs`). Kokoro는 자체 음소 변환기를 쓰므로 토크나이저 파일이 원래 없습니다.

임시 방법은 Kokoro 파일을 링크한 폴더에 아무 `tokenizer.json`이나 하나 넣는 것입니다. 이 파일은 실제로는 쓰이지 않습니다.

```bash
K=~/models/kokoro-82m
mkdir -p $K
for f in config.json kokoro-v1_0.safetensors voices; do
  ln -s ~/.cache/mlxcel/models/mlx-community/Kokoro-82M-bf16/$f $K/$f
done
cp ~/.cache/mlxcel/models/openai/whisper-small/tokenizer.json $K/

mlxcel-server -m $K --port 8081
```

시작할 때 `ERROR ... Missing directory loader for model type: Kokoro` 로그가 찍히지만, 바로 다음 줄에서 TTS 경로로 넘어가므로 무시해도 됩니다.

### STT (Whisper)

```bash
mlxcel-server -m openai/whisper-small --port 8082
```

여기서도 `ERROR ... Whisper is a speech-to-text model ...` 로그가 한 번 나오지만 정상입니다.

## 3. 코드

```python
# /// script
# dependencies = ["openai>=1.60"]
# ///
import time
from openai import OpenAI

tts = OpenAI(base_url="http://localhost:8081/v1", api_key="local")   # Kokoro server
stt = OpenAI(base_url="http://localhost:8082/v1", api_key="local")   # Whisper server

TEXT = ("The quick brown fox jumps over the lazy dog. "
        "Local speech models can run entirely on a laptop.")

# 1. Text to speech: WAV, 24 kHz mono. Voice names are the files in voices/.
for voice in ["af_heart", "bm_lewis"]:
    t = time.perf_counter()
    audio = tts.audio.speech.create(model="kokoro", voice=voice, input=TEXT,
                                    response_format="wav")
    path = f"speech_{voice}.wav"
    audio.write_to_file(path)
    print(f"TTS {voice}: {path} in {time.perf_counter() - t:.2f}s")

    # 2. Speech to text: send the file back to Whisper.
    t = time.perf_counter()
    with open(path, "rb") as f:
        text = stt.audio.transcriptions.create(model="whisper", file=f, language="en").text
    print(f"STT        : {text.strip()!r} in {time.perf_counter() - t:.2f}s")
```

## 실행 결과

whisper-small:

```text
TTS af_heart: speech_af_heart.wav in 0.31s
STT        : 'The quick brown fox jumps over the lazy dog. Lock-all speech models can run entirely on a laptop.' in 0.38s
TTS bm_lewis: speech_bm_lewis.wav in 0.26s
STT        : 'The quick brown fox jumps over the lazy dog. Larkal speech models can run interrally on a laptop.' in 0.33s
```

같은 음성을 whisper-tiny(148 MB)로 받아쓰면 오류가 훨씬 많습니다.

```text
STT (tiny) : 'The quick brown fox jumps over the lazy dog. Lockle speech models can run into your belly on a laptop.'
STT (tiny) : 'The quick brown fucks jumps over the lazy dog. Larkle-speech models can run enterily on a laptop.'
```

- 생성된 파일: 16비트 PCM, 24 kHz 모노, 6.7초 분량을 약 0.3초에 생성
- 두 Whisper 모두 "Local"을 "Lock-all / Larkal"로 들었습니다. Whisper보다는 Kokoro가 이 단어를 어색하게 발음한 것으로 보입니다.

## 주의

- **Whisper 입력은 일반 PCM WAV만** 받습니다. MP3·M4A는 먼저 변환해야 하는데, macOS의 `afconvert`가 만드는 WAV는 확장 형식(WAVE_FORMAT_EXTENSIBLE)이라 그대로 보내면 거부됩니다.

  ```text
  {"error":{"message":"audio model inference failed: WAV decode failed: audio clip 0 is corrupt:
   unsupported WAV format 65534/16-bit","type":"server_error","code":null}}
  ```

  `afconvert`로 바꾼 뒤 Python 3.12 이상의 `wave` 모듈로 한 번 더 일반 PCM으로 쓰면 됩니다(확인함).

  ```python
  # to_pcm_wav.py: rewrite any WAV (including WAVE_FORMAT_EXTENSIBLE) as plain 16-bit PCM.
  import sys
  import wave

  with wave.open(sys.argv[1], "rb") as src:
      params, frames = src.getparams(), src.readframes(src.getnframes())
  with wave.open(sys.argv[2], "wb") as dst:
      dst.setparams(params)
      dst.writeframes(frames)
  ```

  ```bash
  afconvert -f WAVE -d LEI16@16000 -c 1 meeting.m4a tmp.wav
  uv run --python 3.12 to_pcm_wav.py tmp.wav meeting.wav
  ```

  `ffmpeg`가 있다면 `ffmpeg -i meeting.mp3 -ar 16000 -ac 1 meeting.wav`도 일반적인 방법이지만, 이 검증 환경에는 ffmpeg가 없어 확인하지 못했습니다.
- Whisper는 양자화되지 않은(fp16/f32) 체크포인트만 됩니다.
- Kokoro의 발음 변환은 **미국 영어 전용**입니다. 한국어 텍스트는 제대로 읽지 못합니다.
- `voice`에 없는 이름을 넣으면 오류 없이 `af_heart`로 바뀝니다. 목소리 목록은 `voices/` 폴더의 파일 이름입니다.

## 응용

- 회의 녹음 WAV → Whisper → [04번](04-structured-extraction.md)으로 `{결정사항, 할일}` 추출
- 에이전트([03번](03-tool-agent.md))의 답을 Kokoro로 읽어 주는 음성 비서
