# SoundText4Agent_config.py
# Configuration module for SoundText4Agent, supporting YAML file overrides.
# Place a config.yaml file in the same directory to customize settings.

import os
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Please run: pip install pyyaml")
    sys.exit(1)

# ----------------------------------------------------------------------
# Default TTS Configuration (Text-to-Speech)
# ----------------------------------------------------------------------
DEFAULT_TTS_CONFIG = {
    "engine": "matcha",
    "model_path": "./model/matcha-icefall-zh-en/model-steps-3.onnx",
    "vocoder": "./model/vocos-16khz-univ.onnx",
    "lexicon": "./model/matcha-icefall-zh-en/lexicon.txt",
    "tokens": "./model/matcha-icefall-zh-en/tokens.txt",
    "data_dir": "./model/matcha-icefall-zh-en/espeak-ng-data",
    "num_threads": 1,
    "play_audio": True,
}

# ----------------------------------------------------------------------
# Default STT Configuration (Speech-to-Text, including Wake Word, VAD, Whisper)
# ----------------------------------------------------------------------
DEFAULT_STT_CONFIG = {
    # --- Wake word model (Keyword Spotter) ---
    "tokens": "./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt",
    "encoder": "./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-8-left-64.onnx",
    "decoder": "./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
    "joiner": "./model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/joiner-epoch-13-avg-2-chunk-8-left-64.onnx",
    "keywords_file": "./keywords.txt",
    "keywords_threshold": 0.25,
    # --- VAD model ---
    "vad_model": "./model/vad/silero_vad.onnx",
    "vad_threshold": 0.7,
    "min_silence_duration": 0.6,
    "min_speech_duration": 0.3,
    # --- Whisper ASR model ---
    "whisper_encoder": "./model/whisper/base-encoder.onnx",
    "whisper_decoder": "./model/whisper/base-decoder.onnx",
    "whisper_tokens": "./model/whisper/base-tokens.txt",
    "whisper_language": "",          # empty = auto-detect
    "whisper_task": "transcribe",
    "whisper_tail_paddings": 50,     # 50 for English, 300 for Chinese
    # --- System parameters ---
    "sample_rate": 16000,
    "chunk_duration": 0.1,
    "active_timeout": 10.0,
}

# Combined default config
DEFAULT_CONFIG = {
    "tts": DEFAULT_TTS_CONFIG,
    "stt": DEFAULT_STT_CONFIG,
}


def load_config(yaml_path=None):
    """
    Load configuration from a YAML file. If no path is given, look for
    'config.yaml' in the same directory as this module.
    Returns a dictionary with keys 'tts' and 'stt'.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    config = DEFAULT_CONFIG.copy()

    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            # Merge user config with defaults (top-level keys only)
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            print(f"Loaded configuration from {yaml_path}")
        except Exception as e:
            print(f"Error loading {yaml_path}: {e}. Using default configuration.")
    else:
        print(f"Config file {yaml_path} not found. Using default configuration.")

    return config


# Load configuration once at module import
_config = load_config()
TTS_CONFIG = _config.get("tts", DEFAULT_TTS_CONFIG)
STT_CONFIG = _config.get("stt", DEFAULT_STT_CONFIG)

# Optionally, expose the combined config
CONFIG = _config


# ----------------------------------------------------------------------
# Example config.yaml file content (create this file to customize):
#
# tts:
#   model_path: "/custom/path/model.onnx"
#   num_threads: 2
#
# stt:
#   keywords_threshold: 0.15
#   active_timeout: 15.0
#   whisper_language: "zh"
# ----------------------------------------------------------------------