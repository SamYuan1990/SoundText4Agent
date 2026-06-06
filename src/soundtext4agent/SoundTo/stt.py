"""
File path: src/soundtext4agent/SoundTo/stt.py
Description: Sound to text mgr, 融合唤醒词、VAD和离线Whisper ASR的语音交互系统

Feature and design goals：
- A Singleton mgr owns vad, wake up and asr.
- Runs a background thread that detect voice from microphone, and send any detect text into ControlPlane and set ControlPlane to listen mode
ref
    from soundtext4agent.control_plane import ControlPlane
    cp = ControlPlane.get_instance()
    cp.switch_to_listen()
    cp.enqueue_listen_text("Hello from listen mode")

- 平时只进行轻量的唤醒词检测。 after wake up, cp.switch_to_listen()
- 唤醒后进入活跃模式，在活跃期间，系统将使用VAD检测语音片段，并使用 OfflineRecognizer.from_whisper 对其进行识别。
- For text as Recognizer result, send to ControlPlane via cp.enqueue_listen_text(text)
- 若超过指定时间(active-timeout)无有效语音输入，则自动退出活跃模式，回到唤醒词监听状态。
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

# Attempt to import the shared configuration

from soundtext4agent.config import STT_CONFIG
from soundtext4agent.control_plane import ControlPlane

def assert_file_exists(filename: str):
    """检查文件是否存在"""
    if not Path(filename).is_file():
        raise FileNotFoundError(f"File not found: {filename}")


def create_keyword_spotter(cfg: dict):
    """创建唤醒词检测器"""
    return sherpa_onnx.KeywordSpotter(
        tokens=cfg['tokens'],
        encoder=cfg['encoder'],
        decoder=cfg['decoder'],
        joiner=cfg['joiner'],
        num_threads=cfg.get('num_threads', 1),
        max_active_paths=cfg.get('max_active_paths', 4),
        keywords_file=cfg['keywords_file'],
        keywords_score=cfg.get('keywords_score', 1.0),
        keywords_threshold=cfg['keywords_threshold'],
        num_trailing_blanks=cfg.get('num_trailing_blanks', 1),
        provider=cfg.get('provider', 'cpu'),
    )


def create_vad(cfg: dict):
    """创建 VAD 检测器"""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = cfg['vad_model']
    config.silero_vad.threshold = cfg['vad_threshold']
    config.silero_vad.min_silence_duration = cfg['min_silence_duration']
    config.silero_vad.min_speech_duration = cfg['min_speech_duration']
    config.sample_rate = cfg['sample_rate']
    return sherpa_onnx.VoiceActivityDetector(config)


def create_whisper_asr(cfg: dict):
    """创建离线 Whisper 识别器"""
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=cfg['whisper_encoder'],
        decoder=cfg['whisper_decoder'],
        tokens=cfg['whisper_tokens'],
        num_threads=cfg.get('num_threads', 1),
        language=cfg.get('whisper_language', ''),
        task=cfg.get('whisper_task', 'transcribe'),
        tail_paddings=cfg.get('whisper_tail_paddings', 50),
    )


class SoundToTextManager:
    """
    单例语音交互管理器：包含唤醒词、VAD、离线Whisper ASR。
    在后台线程中运行，检测语音并将识别文本发送到 ControlPlane。
    默认配置来自 SoundText4Agent_config.STT_CONFIG，可通过关键字参数覆盖。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.logger = logging.getLogger(__name__)

        # 从全局 STT_CONFIG 加载默认值，再用传入的 kwargs 覆盖
        cfg = STT_CONFIG.copy() if STT_CONFIG else {}
        cfg.update(kwargs)

        # 唤醒词模型参数
        self.tokens = cfg.get('tokens')
        self.encoder = cfg.get('encoder')
        self.decoder = cfg.get('decoder')
        self.joiner = cfg.get('joiner')
        self.keywords_file = cfg.get('keywords_file')
        self.num_threads = cfg.get('num_threads', 1)
        self.provider = cfg.get('provider', 'cpu')
        self.max_active_paths = cfg.get('max_active_paths', 4)
        self.num_trailing_blanks = cfg.get('num_trailing_blanks', 1)
        self.keywords_score = cfg.get('keywords_score', 1.0)
        self.keywords_threshold = cfg.get('keywords_threshold', 0.25)

        # VAD 参数
        self.vad_model = cfg.get('vad_model')
        self.vad_threshold = cfg.get('vad_threshold', 0.7)
        self.min_silence_duration = cfg.get('min_silence_duration', 0.6)
        self.min_speech_duration = cfg.get('min_speech_duration', 0.3)

        # 离线 Whisper ASR 参数
        self.whisper_encoder = cfg.get('whisper_encoder')
        self.whisper_decoder = cfg.get('whisper_decoder')
        self.whisper_tokens = cfg.get('whisper_tokens')
        self.whisper_language = cfg.get('whisper_language', '')
        self.whisper_task = cfg.get('whisper_task', 'transcribe')
        self.whisper_tail_paddings = cfg.get('whisper_tail_paddings', 50)

        # 系统参数
        self.sample_rate = cfg.get('sample_rate', 16000)
        self.chunk_duration = cfg.get('chunk_duration', 0.1)
        self.active_timeout = cfg.get('active_timeout', 10.0)

        # 组件占位
        self.kw_spotter = None
        self.vad = None
        self.whisper_asr = None

        # 线程控制
        self._thread = None
        self._stop_event = threading.Event()
        self.logger.debug("SoundToTextManager instance initialized with config: %s", cfg)

    def _init_components(self):
        """加载并初始化所有模型组件。"""
        required_files = [
            self.tokens, self.encoder, self.decoder, self.joiner, self.keywords_file,
            self.vad_model, self.whisper_encoder, self.whisper_decoder, self.whisper_tokens
        ]
        for f in required_files:
            assert_file_exists(f)

        # 构建配置字典以便传递给工厂函数
        cfg = {
            'tokens': self.tokens,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'joiner': self.joiner,
            'keywords_file': self.keywords_file,
            'num_threads': self.num_threads,
            'provider': self.provider,
            'max_active_paths': self.max_active_paths,
            'num_trailing_blanks': self.num_trailing_blanks,
            'keywords_score': self.keywords_score,
            'keywords_threshold': self.keywords_threshold,
            'vad_model': self.vad_model,
            'vad_threshold': self.vad_threshold,
            'min_silence_duration': self.min_silence_duration,
            'min_speech_duration': self.min_speech_duration,
            'sample_rate': self.sample_rate,
            'whisper_encoder': self.whisper_encoder,
            'whisper_decoder': self.whisper_decoder,
            'whisper_tokens': self.whisper_tokens,
            'whisper_language': self.whisper_language,
            'whisper_task': self.whisper_task,
            'whisper_tail_paddings': self.whisper_tail_paddings,
        }
        self.kw_spotter = create_keyword_spotter(cfg)
        self.vad = create_vad(cfg)
        self.whisper_asr = create_whisper_asr(cfg)
        self.logger.info("All components loaded successfully")

    def start(self, block=False):
        """
        启动后台语音处理线程。
        :param block: 如果为 True，阻塞当前线程直到后台线程结束。
        """
        if self.kw_spotter is None:
            self._init_components()
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("Manager thread is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="stt-manager")
        self._thread.start()
        self.logger.info("SoundToTextManager thread started")
        if block:
            try:
                self._thread.join()
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """停止后台线程并释放资源。"""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self.logger.warning("Manager thread did not stop in time, forcing exit")
        self._thread = None
        self.logger.info("SoundToTextManager thread stopped")

    def _run(self):
        """后台线程主循环：音频采集、状态机、识别输出。"""
        cp = ControlPlane.get_instance()
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        MODE_WAKE = 0
        MODE_ACTIVE = 1
        mode = MODE_WAKE
        kw_stream = self.kw_spotter.create_stream()
        last_valid_speech_time = None

        try:
            with sd.InputStream(channels=1, dtype="float32", samplerate=self.sample_rate) as s:
                self.logger.info("Microphone stream opened, waiting for wake word...")
                while not self._stop_event.is_set():
                    samples, _ = s.read(chunk_samples)
                    samples = samples.reshape(-1)

                    if mode == MODE_WAKE:
                        kw_stream.accept_waveform(self.sample_rate, samples)
                        while self.kw_spotter.is_ready(kw_stream):
                            self.kw_spotter.decode_stream(kw_stream)
                            result = self.kw_spotter.get_result(kw_stream)
                            if result:
                                self.logger.info("Wake word detected: %s", result)
                                cp.switch_to_listen()
                                self.logger.info("Entering active mode (timeout: %s s)", self.active_timeout)
                                mode = MODE_ACTIVE
                                self.vad.reset()
                                last_valid_speech_time = time.time()
                                kw_stream = self.kw_spotter.create_stream()
                                break
                    else:  # MODE_ACTIVE
                        self.vad.accept_waveform(samples)
                        while not self.vad.empty():
                            seg = self.vad.front
                            if seg.samples:
                                speech_audio = np.array(seg.samples, dtype=np.float32)
                                if len(speech_audio) >= 0.5 * self.sample_rate:
                                    self.logger.debug("Processing speech segment of length %.2f s",
                                                      len(speech_audio) / self.sample_rate)
                                    stream = self.whisper_asr.create_stream()
                                    stream.accept_waveform(self.sample_rate, speech_audio.tolist())
                                    self.whisper_asr.decode_stream(stream)
                                    text = stream.result.text.strip()
                                    if text:
                                        self.logger.info("ASR result: %s", text)
                                        cp.enqueue_listen_text(text)
                                        last_valid_speech_time = time.time()
                                    else:
                                        self.logger.debug("No valid text recognized")
                                else:
                                    self.logger.debug("Speech segment too short, ignored")
                            else:
                                self.logger.debug("VAD segment has no samples")
                            self.vad.pop()

                        # 超时检查
                        if last_valid_speech_time is not None:
                            if time.time() - last_valid_speech_time > self.active_timeout:
                                self.logger.info("Active mode timeout, returning to wake mode")
                                mode = MODE_WAKE
                                kw_stream = self.kw_spotter.create_stream()
                                self.vad.reset()
                                last_valid_speech_time = None

                    time.sleep(0.005)
        except Exception as e:
            self.logger.exception("Error in audio processing loop: %s", e)
        finally:
            self.logger.info("Audio processing loop exited")


"""
Example usage:
    # 在其他文件中引用
    from soundtext4agent.SoundTo.stt import SoundToTextManager

    # 获取单例（自动使用 SoundText4Agent_config.STT_CONFIG 的默认值）
    manager = SoundToTextManager()

    # 或者覆盖某些参数
    manager = SoundToTextManager(active_timeout=15.0)

    # 启动后台语音处理（非阻塞）
    manager.start()

    # 之后可以执行其他任务，语音识别结果会自动发送到 ControlPlane
    import time
    time.sleep(60)

    # 停止后台线程
    manager.stop()
"""