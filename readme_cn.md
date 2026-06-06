<h1 align="center">
  🎙️ SoundText4Agent
</h1>
<p align="center">
  <strong>语音交互中间件 · 为智能体而生</strong><br>
  <em>融合唤醒词、语音识别、语音合成与唇形同步的一站式服务</em>
</p>

---

## ✨ 特性

- **🎤 智能语音输入 (STT)**
  - 轻量级**唤醒词检测**，平时不消耗过多资源
  - 唤醒后自动进入**活跃监听模式**，利用 VAD 切分语音片段
  - 集成 **Whisper 离线识别**，高精度转写
  - 超时自动休眠，回到低功耗唤醒等待状态

- **🗣️ 语音合成 (TTS)**
  - 文本转 WAV 音频文件
  - 支持单例重用与配置驱动

- **👄 唇形同步 (TTL)**
  - 文本 → 音素时间线（`phoneme, start_sec, end_sec`）
  - 基于 espeak-ng + piper‑phonemize，为具身智能提供口型数据
  - 失败时自动降级为启发式时间线生成

- **🧠 控制平面 (ControlPlane)**
  - 全局单例，统一管理 `listen` / `speak` 状态
  - 两个队列：
    - `listen_text` – 接收 STT 识别结果
    - `speak_text` – 存放待合成的文本
  - 队列方法：`enqueue_listen_text`、`dequeue_listen_text`、`enqueue_speak_text`、`dequeue_speak_text`
  - 状态切换时自动清理监听队列，保证上下文干净

- **📦 多种交付方式（规划中）**
  - Python SDK 直接集成
  - RESTful API 服务
  - Function Call / MCP 协议供大模型调用

---

## 📦 安装

**环境要求**：Python 3.10+，推荐使用 [`uv`](https://github.com/astral-sh/uv) 管理依赖。

```bash
# 1. 同步依赖
uv sync

# 2. 下载模型文件到 model/ 目录（从对应 model space 获取）
#    请确保 model/ 下包含 sherpa-onnx-kws-zipformer 等所需模型

# 3. 生成唤醒词 tokens
python ./src/soundtext4agent/SoundTo/text2token.py \
  --tokens ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
  --tokens-type phone+ppinyin \
  --lexicon ./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/en.phone \
  --text ./keywords_rwa.txt \
  --output ./keywords.txt
```

> 📁 推荐将自定义配置文件 `config.yaml` 放在 `src/soundtext4agent/` 下，程序会自动读取并覆盖默认值。

---

## 🚀 快速开始

```python
from soundtext4agent.control_plane import ControlPlane
from soundtext4agent.SoundTo.stt import SoundToTextManager     # STT 管理器
from soundtext4agent.TextTo.ttmgr import TextToManager        # TTS + TTL 管理器

# 1. 控制平面（默认 listen 状态）
cp = ControlPlane.get_instance()

# 2. 启动 TTS 管理器（配置并启动）
tts = TextToManager.configure()
tts.start()

# 3. 启动 STT 管理器
stt = SoundToTextManager()
stt.start()                     # 开始监听麦克风，包含唤醒词

# 此时系统已就绪，唤醒后会进行语音识别，结果进入 listen 队列。
# 你可以从 listen 队列取出文本，再送入 speak 队列让 TTS 合成。
```

> 💡 完整的回声（语音进、语音出）示例请查看下方的 **回音示例** 章节。

---

## 🎤 回音示例（语音输入 → 语音输出）

项目中的 `example.py` 文件展示了完整的回声管线：

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
    # 1. 初始化 TTS
    tts_manager = TextToManager.configure()
    tts_manager.start()

    # 2. 初始化 STT
    stt_manager = SoundToTextManager()
    stt_manager.start()

    cp = ControlPlane.get_instance()

    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            # 等待 STT 识别出的文本
            recognised_text = cp.dequeue_listen_text()
            if recognised_text:
                logger.info(f"识别结果: {recognised_text}")
                # 回声：将文本送入 speak 队列，让 TTS 合成并播放
                cp.enqueue_speak_text(recognised_text)
    except KeyboardInterrupt:
        logger.info("用户中断。")
    finally:
        stt_manager.stop()
        tts_manager.stop()
        logger.info("回音演示结束。")

if __name__ == "__main__":
    echo_loop(duration_seconds=60)
```

**工作流程：**
1. `SoundToTextManager` 后台线程持续监听唤醒词，唤醒后通过 VAD + Whisper 识别语音，并将识别文本通过 `enqueue_listen_text` 送入控制平面的 `listen_text` 队列。
2. 示例循环调用 `cp.dequeue_listen_text()` 取出识别结果。
3. 立即将文本通过 `cp.enqueue_speak_text()` 送入 `speak_text` 队列。
4. `TextToManager` 后台线程不断从 `speak_text` 队列取文本，并行合成音频与唇形数据，并输出结果。

---

## 📐 架构概览

```
┌─────────────────────────────────────────────┐
│                  ControlPlane                 │
│  状态: listen | speak                         │
│  队列: listen_text (入), speak_text (出)      │
└───────────────┬──────────────┬──────────────┘
                │              │
    ┌───────────▼────┐  ┌─────▼──────────────┐
    │  SoundTo (STT)  │  │ TextTo (TTS + TTL) │
    │  · 唤醒词       │  │  · TTS → wav       │
    │  · VAD          │  │  · TTL → lip token │
    │  · Whisper ASR  │  │  · 结果队列        │
    └────────────────┘  └────────────────────┘
                │              │
          麦克风输入        输出音频 + 口型时间线
```

- **SoundTo/** – 融合唤醒词、VAD 和离线 Whisper，后台持续工作，识别结果通过 `ControlPlane.enqueue_listen_text()` 入队。
- **TextTo/** – 单例管理器统一调度 TTS 和 TTL，从 speak 队列取文本（`dequeue_speak_text()`），并行处理，结果放入可消费的结果队列。
- **ControlPlane** – 无锁单例状态机，提供两个队列作为数据总线，隔离输入与输出方向。
- **config.py** – 全局配置模块，支持 YAML 文件覆盖，方便部署时调整参数。

---

## 🗺️ 路线图

以下功能已列入开发计划，欢迎贡献：

- [ ] **REST API / MCP 服务** – 将核心能力暴露为网络协议
- [ ] **流式 TTS** – 边生成边播放，降低首字延迟
- [ ] **音色克隆** – 支持说话人嵌入，自定义语音
- [ ] **控制平面线程安全** – 增加锁机制以支持多线程并发
- [ ] **队列容量限制** – 防止内存无限增长
- [ ] **动态换模** – 不重启情况下切换 TTS/ASR 模型
- [ ] **可观测性** – 集成 OpenTelemetry / Prometheus 指标
- [ ] **完整测试覆盖** – 单元测试、集成测试

---

## 🤝 贡献指南

我们采用 `uv` 进行依赖管理，请遵守以下编码约定：

```bash
# 运行时依赖
uv add requests

# 开发依赖
uv add --dev ruff pytest mypy pre-commit
```

- 代码风格由 `ruff` 自动检查
- 提交前请运行 `pytest` 与 `mypy`
- 新功能请尽量保持单例模式与控制平面的设计哲学

---

## 📄 许可证

本项目基于 [Apache 2.0 许可证](./LICENSE) 开源。

---

<p align="center">Made with ❤️ for the agent era</p>