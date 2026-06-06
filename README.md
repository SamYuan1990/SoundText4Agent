# SoundText4Agent

STT and TTS service for agent

## Features
- Status switch between listen and speak as control plane
- Listen supports
- vad
- wake words
- STT
- Speak supports
- wav
- lip shape token for embodied intelligence

- Deliver via
- SDK usage
- Restful api
- function call
- MCP

## Install via

`uv sync`
```
python ./src/soundtext4agent/SoundTo/text2token.py \
  --tokens ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
  --tokens-type phone+ppinyin \
  --lexicon ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/en.phone \
  --text ./keywords_rwa.txt \
  --output ./keywords.txt
```

download model from model space and input at model folder

## Coding rules

```
# run time dep（例如 requests）
uv add requests

# dev dep
uv add --dev ruff pytest mypy pre-commit
```