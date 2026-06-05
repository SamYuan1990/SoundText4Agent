# SoundText4Agent

STT and TTS service for agent

## Features
- Status switch between listen and speak as control plane
- Listen supports
- [ ] vad
- [ ] weak words
- [ ] STT
- Speak supports
- [ ] wav
- [ ] lip shape token for embodied intelligence

- Deliver via
- SDK usage
- Restful api
- function call
- MCP

## Install via

`uv sync`

## Coding rules

```
# run time dep（例如 requests）
uv add requests

# dev dep
uv add --dev ruff pytest mypy pre-commit
```