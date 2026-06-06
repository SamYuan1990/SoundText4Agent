<h1 align="center">
  🎙️ SoundText4Agent
</h1>
<p align="center">
  <strong>Voice Interaction Middleware · Built for Agents</strong><br>
  <em>A unified service combining wake word detection, speech recognition, speech synthesis, and lip‑sync</em>
</p>

---

## ✨ Features

- **🎤 Intelligent Speech Input (STT)**
  - Lightweight **wake word detection** with low standby resource usage
  - Automatically enters **active listening mode** upon wake‑up, using VAD to segment speech
  - Integrated **offline Whisper recognition** for high‑accuracy transcription
  - Auto‑sleeps after a configurable timeout, returning to low‑power wake word waiting

- **🗣️ Speech Synthesis (TTS)**
  - Converts text to WAV audio files
  - Singleton reuse and configuration‑driven design

- **👄 Lip‑Sync (TTL)**
  - Converts text into phoneme timelines (`phoneme, start_sec, end_sec`)
  - Powered by espeak‑ng + piper‑phonemize, providing mouth‑shape data for embodied agents
  - Graceful fallback to heuristic timeline generation on failure

- **🧠 Control Plane (ControlPlane)**
  - Global singleton that manages `listen` / `speak` state
  - Two queues:
    - `listen_text` – receives recognised speech from STT
    - `speak_text` – holds text to be spoken by TTS
  - Queue methods: `enqueue_listen_text`, `dequeue_listen_text`, `enqueue_speak_text`, `dequeue_speak_text`
  - Automatically clears the listen queue on state switch, keeping context clean

- **📦 Multiple Delivery Modes (planned)**
  - Direct Python SDK integration
  - RESTful API service
  - Function Call / MCP protocol for LLM‑driven agents

---

## 📦 Installation

**Requirements**: Python 3.10+, dependency management with [`uv`](https://github.com/astral-sh/uv) recommended.

```bash
# 1. Install dependencies
uv sync

# 2. Download required models into the model/ directory (from the corresponding model space)
#    Ensure model/ contains sherpa-onnx-kws-zipformer and any other needed models

# 3. Generate wake word tokens
python ./src/soundtext4agent/SoundTo/text2token.py \
  --tokens ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
  --tokens-type phone+ppinyin \
  --lexicon ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/en.phone \
  --text ./keywords_rwa.txt \
  --output ./keywords.txt
```

> 📁 Place a custom `config.yaml` inside `src/soundtext4agent/` to override default settings.

---

## 🚀 Quick Start

```python
from soundtext4agent.control_plane import ControlPlane
from soundtext4agent.SoundTo.stt import SoundToTextManager     # STT manager
from soundtext4agent.TextTo.ttmgr import TextToManager        # TTS + TTL manager

# 1. Control plane (default state: listen)
cp = ControlPlane.get_instance()

# 2. Start TTS manager (configure + start)
tts = TextToManager.configure()
tts.start()

# 3. Start STT manager
stt = SoundToTextManager()
stt.start()                     # begins microphone listening with wake word

# Now the system is live – it will wake, recognise speech, and fill the listen queue.
# You can process recognised text and feed it into TTS via the speak queue.
```

> 💡 For a complete echo (speech‑to‑speech) example, see the **Echo Demo** section below.

---

## 🎤 Echo Demo (speech‑in, speech‑out)

The file `example.py` (provided in the repository) demonstrates a full echo pipeline:

```python
import time
import logging
from soundtext4agent.SoundTo.stt import SoundToTextManager
from soundtext4agent.TextTo.ttmgr import TextToManager
from soundtext4agent.control_plane import ControlPlane

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("echo_demo")

def echo_loop(duration_seconds: int = 60):
    # 1. Initialise TTS
    tts_manager = TextToManager.configure()
    tts_manager.start()

    # 2. Initialise STT
    stt_manager = SoundToTextManager()
    stt_manager.start()

    cp = ControlPlane.get_instance()

    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            # Wait for a recognised phrase from STT
            recognised_text = cp.dequeue_listen_text()
            if recognised_text:
                logger.info(f"Recognised: {recognised_text}")
                # Echo: send text to TTS via the speak queue
                cp.enqueue_speak_text(recognised_text)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stt_manager.stop()
        tts_manager.stop()
        logger.info("Echo demonstration finished.")

if __name__ == "__main__":
    echo_loop(duration_seconds=60)
```

**How it works:**
1. `SoundToTextManager` runs a background thread that detects the wake word, then captures speech via VAD + Whisper, and pushes the recognised text into the ControlPlane’s `listen_text` queue (via `enqueue_listen_text`).
2. The demo loop calls `cp.dequeue_listen_text()` to fetch that text.
3. It immediately sends the text into the `speak_text` queue with `cp.enqueue_speak_text()`.
4. `TextToManager` (which runs its own background thread) continuously pulls from the `speak_text` queue, synthesises audio + lip‑sync data, and makes the results available.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────┐
│                  ControlPlane                 │
│  State: listen | speak                        │
│  Queues: listen_text (in), speak_text (out)   │
└───────────────┬──────────────┬──────────────┘
                │              │
    ┌───────────▼────┐  ┌─────▼──────────────┐
    │  SoundTo (STT)  │  │ TextTo (TTS + TTL) │
    │  · Wake word    │  │  · TTS → wav       │
    │  · VAD          │  │  · TTL → lip token │
    │  · Whisper ASR  │  │  · Result queue    │
    └────────────────┘  └────────────────────┘
                │              │
       Microphone input   Output audio + lip timeline
```

- **SoundTo/** – Combines wake word detection, VAD, and offline Whisper. Runs continuously in the background, pushing recognised text to `ControlPlane.enqueue_listen_text()`.
- **TextTo/** – Singleton manager that orchestrates TTS and TTL. Pulls text from the speak queue (`dequeue_speak_text()`), processes both in parallel, and places results into a consumable results queue.
- **ControlPlane** – Lock‑free singleton state machine that provides two queues as a data bus, isolating input and output flows.
- **config.py** – Global configuration module with YAML override support for easy deployment tuning.

---

## 🗺️ Roadmap

The following features are on the development plan. Contributions are welcome:

- [ ] **REST API / MCP Service** – Expose core capabilities via network protocols
- [ ] **Streaming TTS** – Incremental generation to reduce time‑to‑first‑audio
- [ ] **Voice Cloning** – Speaker embedding input for custom voices
- [ ] **Thread‑safe Control Plane** – Add locking for concurrent access
- [ ] **Queue Size Limits** – Prevent unbounded memory growth
- [ ] **Dynamic Model Swap** – Switch TTS/ASR models without restart
- [ ] **Observability** – Integrate OpenTelemetry / Prometheus metrics
- [ ] **Test Coverage** – Unit tests and integration tests

---

## 🤝 Contributing

We use `uv` for dependency management. Please follow these coding conventions:

```bash
# Run‑time dependencies
uv add requests

# Dev dependencies
uv add --dev ruff pytest mypy pre-commit
```

- Code style is enforced by `ruff`
- Run `pytest` and `mypy` before committing
- New features should align with the singleton pattern and control‑plane design philosophy

---

## 📄 License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

<p align="center">Made with ❤️ for the agent era</p>