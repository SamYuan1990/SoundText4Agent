"""
file path: src/soundtext4agent/TextTo/tts.py
description: text to speech service

Features and design goals
- Singleton: only one instance of speech is created and reused.
- Receives text and outputs as soundfile format wave.
- Has a config and play the soundfile format wave as default option.

Missing features (not yet implemented, kept for future consideration)
- Streaming TTS (generate audio incrementally).
- Voice cloning / speaker embedding input.
- Dynamic model hot‑swap without restart.
- Comprehensive error handling and fallback (e.g., multiple model providers).
- Metrics and tracing integration (OpenTelemetry helpers are commented out as placeholders).
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import sherpa_onnx

# Attempt to import sounddevice for audio playback.
try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except ImportError:
    sd = None
    PLAYBACK_AVAILABLE = False

# from opentelemetry import trace
# from opentelemetry import metrics
# meter = metrics.get_meter(__name__)
# tracer = trace.get_tracer(__name__)


class TextToSpeech:
    """
    A singleton wrapper around sherpa-onnx OfflineTTS.

    Usage:
        config = {
            "engine": "vits",
            "model_path": "...",
            "tokens": "...",
            "data_dir": "...",
            "provider": "cpu",
            "num_threads": 2,
            "play_audio": True,   # default True: auto-play after speak()
        }
        tts = TextToSpeech.get_instance(config)
        tts.speak("Hello world")   # generates and plays audio
        samples, rate = tts.generate("Hello world")  # raw samples
    """

    _instance: Optional["TextToSpeech"] = None
    _initialized: bool = False

    def __new__(cls) -> "TextToSpeech":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # __init__ is called every time an instance is created.
        # Actual engine initialization is deferred to _initialize_engine()
        # which is called only once via get_instance().
        pass

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> "TextToSpeech":
        """
        Return the singleton instance, initializing it on the first call.

        Args:
            config: Dictionary with engine configuration. Required on first call.
                    Supported keys:
                        - engine (str): one of "vits", "matcha", "kokoro", "kitten"
                        - model_path (str): path to the main model file
                        - tokens (str): path to tokens.txt
                        - data_dir (str): path to espeak‑ng data directory (optional)
                        - lexicon (str): path to lexicon.txt (optional)
                        - vocoder (str): path to vocoder model (for matcha)
                        - voices (str): path to voices.bin (for kokoro/kitten)
                        - provider (str): "cpu" (default), "cuda", "coreml"
                        - num_threads (int): number of threads (default 1)
                        - debug (bool): enable debug output (default False)
                        - rule_fsts (str): comma separated list of fst files (optional)
                        - max_num_sentences (int): batch size limit (default 1)
                        - play_audio (bool): whether to play audio after generation in speak() (default True)
        Returns:
            TextToSpeech singleton
        """
        instance = cls()
        if not cls._initialized:
            if config is None:
                raise ValueError(
                    "Configuration must be provided when initializing the TTS service for the first time."
                )
            instance._initialize_engine(config)
            cls._initialized = True
        return instance

    def _initialize_engine(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TextToSpeech service...")

        engine_type = config.get("engine", "").lower()
        if engine_type not in ("vits", "matcha", "kokoro", "kitten"):
            raise ValueError(f"Unsupported engine type: {engine_type}")

        # Prepare common args
        provider = config.get("provider", "cpu")
        num_threads = config.get("num_threads", 1)
        debug = config.get("debug", False)
        rule_fsts = config.get("rule_fsts", "")
        max_num_sentences = config.get("max_num_sentences", 1)

        # Build the appropriate model config
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            debug=debug,
            num_threads=num_threads,
            provider=provider,
        )

        if engine_type == "vits":
            model_config.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=config["model_path"],
                tokens=config.get("tokens", ""),
                data_dir=config.get("data_dir", ""),
                lexicon=config.get("lexicon", ""),
            )
        elif engine_type == "matcha":
            model_config.matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=config["model_path"],
                vocoder=config.get("vocoder", ""),
                tokens=config.get("tokens", ""),
                data_dir=config.get("data_dir", ""),
                lexicon=config.get("lexicon", ""),
            )
        elif engine_type == "kokoro":
            model_config.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=config["model_path"],
                voices=config.get("voices", ""),
                tokens=config.get("tokens", ""),
                data_dir=config.get("data_dir", ""),
                lexicon=config.get("lexicon", ""),
            )
        elif engine_type == "kitten":
            model_config.kitten = sherpa_onnx.OfflineTtsKittenModelConfig(
                model=config["model_path"],
                voices=config.get("voices", ""),
                tokens=config.get("tokens", ""),
                data_dir=config.get("data_dir", ""),
            )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=rule_fsts,
            max_num_sentences=max_num_sentences,
        )

        if not tts_config.validate():
            raise RuntimeError("Invalid TTS configuration. Please check the provided model paths and settings.")

        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.engine_type = engine_type
        self.play_audio = config.get("play_audio", True)   # default True
        self.logger.info("TextToSpeech engine initialized successfully. play_audio=%s", self.play_audio)

        # Optional metrics (commented out – real implementation would attach a metric)
        # self.generation_counter = meter.create_counter(
        #     "tts.generation.count",
        #     description="Number of TTS generations"
        # )

    def generate(
        self, text: str, sid: int = 0, speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech and return raw audio samples.

        Args:
            text: The input text to synthesize.
            sid: Speaker ID (for multi‑speaker models).
            speed: Speech speed factor (1.0 = normal, >1 faster, <1 slower).

        Returns:
            A tuple (audio_samples, sample_rate) where
                audio_samples is a 1‑D numpy array of float32 PCM samples,
                sample_rate is the audio sample rate in Hz.
        """
        self.logger.debug("Generating audio for text: '%s' (sid=%d, speed=%.2f)", text[:80], sid, speed)

        # current_span = trace.get_current_span()
        # current_span.set_attribute("text.length", len(text))

        start = time.time()
        audio = self.tts.generate(text, sid=sid, speed=speed)
        elapsed = time.time() - start

        if len(audio.samples) == 0:
            self.logger.error("Generated empty audio. Please check previous error messages.")
            raise RuntimeError("TTS generation produced empty audio.")

        duration = len(audio.samples) / audio.sample_rate
        rtf = elapsed / duration
        self.logger.debug(
            "Generated %.2fs of audio in %.2fs (RTF=%.3f)", duration, elapsed, rtf
        )
        # self.generation_counter.add(1)  # if metric was enabled

        return audio.samples, audio.sample_rate

    def speak(self, text: str, sid: int = 0, speed: float = 1.0, block: bool = True) -> None:
        """
        Generate speech from text and play it immediately (if play_audio is True).

        This is the default high‑level interface for the TTS service. Audio is
        played directly to the default output device. If playback is disabled or
        unavailable, the method will only log the action and optionally save nothing.

        Args:
            text: The text to synthesize and play.
            sid: Speaker ID.
            speed: Speech speed factor.
            block: If True, wait until playback finishes; if False, return immediately.
        """
        samples, sample_rate = self.generate(text, sid=sid, speed=speed)

        if self.play_audio:
            if not PLAYBACK_AVAILABLE:
                self.logger.warning(
                    "Audio playback requested but sounddevice is not installed. "
                    "Consider installing sounddevice: pip install sounddevice"
                )
                self.logger.info("Skipping playback. Use save_wav() to save to file instead.")
                return

            self.logger.debug("Playing audio (duration %.2fs, block=%s)", len(samples)/sample_rate, block)
            sd.play(samples, samplerate=sample_rate)
            if block:
                sd.wait()
        else:
            self.logger.debug("Playback disabled, only generated audio (no output).")

    def save_wav(self, filename: str, text: str, sid: int = 0, speed: float = 1.0) -> None:
        """
        Generate speech and save it directly as a WAV file.

        Args:
            filename: Path to the output .wav file.
            text: Input text.
            sid: Speaker ID.
            speed: Speech speed.
        """
        samples, sample_rate = self.generate(text, sid, speed)
        import soundfile as sf
        sf.write(filename, samples, samplerate=sample_rate, subtype="PCM_16")
        self.logger.info("WAV file saved to %s", filename)


"""
Example usage:
    # 1. First‑time initialization (must be done once)
    config = {
        "engine": "vits",
        "model_path": "./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
        "tokens": "./vits-piper-en_US-amy-low/tokens.txt",
        "data_dir": "./vits-piper-en_US-amy-low/espeak-ng-data",
        "num_threads": 2,
        "play_audio": True,   # default; omit to auto-play
    }
    tts = TextToSpeech.get_instance(config)

    # 2. Generate speech and play it immediately (the default behaviour)
    tts.speak("Hello world", sid=0, speed=1.0)

    # 3. Get raw samples for custom processing
    samples, rate = tts.generate("Hello world", sid=0, speed=1.0)
    # write manually:
    # import soundfile as sf
    # sf.write("output.wav", samples, samplerate=rate, subtype="PCM_16")

    # 4. Convenience method to save directly without playing
    tts.save_wav("greeting.wav", "Hello world")

    # 5. Subsequent calls reuse the same instance (singleton)
    tts2 = TextToSpeech.get_instance()  # no config needed
    assert tts is tts2

    # 6. Disable playback globally for a session (needs reinitialization)
    #    In a real app, you might handle it via config at startup.
    #    Currently you'd need to restart the process with play_audio=False
"""