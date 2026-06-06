#!/usr/bin/env python3
"""
融合唤醒词、VAD和离线Whisper ASR的语音交互系统

功能：
1. 平时只进行轻量的唤醒词检测。
2. 唤醒后进入活跃模式，在活跃期间，系统将使用VAD检测语音片段，并使用
   OfflineRecognizer.from_whisper 对其进行识别。
3. 若超过指定时间(active-timeout)无有效语音输入，则自动退出活跃模式，
   回到唤醒词监听状态。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx


def assert_file_exists(filename: str):
    """检查文件是否存在"""
    if not Path(filename).is_file():
        print(f"File not found: {filename}")
        sys.exit(-1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 唤醒词模型参数 (Keyword Spotter) ---
    parser.add_argument("--tokens", type=str,
                        default="/Users/yuanyi/OpenSource/SoundText4Agent/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt",
                        help="唤醒词模型的 tokens.txt")
    parser.add_argument("--encoder", type=str,
                        default="/Users/yuanyi/OpenSource/SoundText4Agent/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-8-left-64.onnx",
                        help="唤醒词模型 encoder.onnx")
    parser.add_argument("--decoder", type=str,
                        default="/Users/yuanyi/OpenSource/SoundText4Agent/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
                        help="唤醒词模型 decoder.onnx")
    parser.add_argument("--joiner", type=str,
                        default="/Users/yuanyi/OpenSource/SoundText4Agent/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/joiner-epoch-13-avg-2-chunk-8-left-64.onnx",
                        help="唤醒词模型 joiner.onnx")
    parser.add_argument("--keywords-file", type=str,
                        default="/Users/yuanyi/OpenSource/SoundText4Agent/keywords.txt",
                        help="唤醒词关键词文件 keywords.txt")
    parser.add_argument("--num-threads", type=int, default=1, help="线程数")
    parser.add_argument("--provider", type=str, default="cpu", help="推理后端")
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--num-trailing-blanks", type=int, default=1)
    parser.add_argument("--keywords-score", type=float, default=1.0)
    parser.add_argument("--keywords-threshold", type=float, default=0.25)

    # --- VAD 参数 ---
    parser.add_argument("--vad-model", type=str,
                        default="/Users/yuanyi/OpenSource/i18n-agent-action/App/storage/data/vad/silero_vad.onnx",
                        help="Silero VAD 模型路径")
    parser.add_argument("--vad-threshold", type=float, default=0.7,
                        help="VAD 阈值 (0-1)，越高越不灵敏（推荐0.5-0.8）")
    parser.add_argument("--min-silence-duration", type=float, default=0.6,
                        help="静音多少秒后判定语音结束")
    parser.add_argument("--min-speech-duration", type=float, default=0.3,
                        help="最短有效语音长度（秒），低于此值忽略")

    # --- 离线 Whisper ASR 参数 ---
    parser.add_argument("--whisper-encoder", type=str,
                        default="/Users/yuanyi/OpenSource/i18n-agent-action/App/storage/data/whisper/base-encoder.onnx",
                        help="Whisper encoder.onnx")
    parser.add_argument("--whisper-decoder", type=str,
                        default="/Users/yuanyi/OpenSource/i18n-agent-action/App/storage/data/whisper/base-decoder.onnx",
                        help="Whisper decoder.onnx")
    parser.add_argument("--whisper-tokens", type=str,
                        default="/Users/yuanyi/OpenSource/i18n-agent-action/App/storage/data/whisper/base-tokens.txt",
                        help="Whisper tokens.txt")
    parser.add_argument("--whisper-language", type=str, default="",
                        help="识别语言(如zh/en)，留空自动检测")
    parser.add_argument("--whisper-task", type=str, default="transcribe",
                        help="任务: transcribe 或 translate")
    parser.add_argument("--whisper-tail-paddings", type=int, default=50,
                        help="尾部填充帧数(英文推荐50，中文推荐300)")

    # --- 系统参数 ---
    parser.add_argument("--sample-rate", type=int, default=16000, help="音频采样率")
    parser.add_argument("--chunk-duration", type=float, default=0.1,
                        help="每次从麦克风读取的音频块长度（秒）")
    parser.add_argument("--active-timeout", type=float, default=10.0,
                        help="唤醒后无有效语音超时时间（秒）")

    return parser.parse_args()


def create_keyword_spotter(args):
    """创建唤醒词检测器"""
    return sherpa_onnx.KeywordSpotter(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        max_active_paths=args.max_active_paths,
        keywords_file=args.keywords_file,
        keywords_score=args.keywords_score,
        keywords_threshold=args.keywords_threshold,
        num_trailing_blanks=args.num_trailing_blanks,
        provider=args.provider,
    )


def create_vad(args):
    """创建 VAD 检测器"""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.vad_model
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence_duration
    config.silero_vad.min_speech_duration = args.min_speech_duration
    config.sample_rate = args.sample_rate
    return sherpa_onnx.VoiceActivityDetector(config)


def create_whisper_asr(args):
    """创建离线 Whisper 识别器"""
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.whisper_encoder,
        decoder=args.whisper_decoder,
        tokens=args.whisper_tokens,
        num_threads=args.num_threads,
        language=args.whisper_language,
        task=args.whisper_task,
        tail_paddings=args.whisper_tail_paddings,
    )


def process_speech_segment(segment: np.ndarray, sample_rate: int, asr):
    """使用 Whisper 识别音频片段"""
    min_len = int(0.5 * sample_rate)  # 最短0.5秒
    if len(segment) < min_len:
        print("(语音段过短，已忽略)")
        return False  # 返回 False 表示未处理有效语音
    stream = asr.create_stream()
    stream.accept_waveform(sample_rate, segment.tolist())
    asr.decode_stream(stream)
    result = stream.result.text
    if result and result.strip():
        print(f"识别结果: {result}")
        return True
    else:
        print("(未识别到有效文本)")
        return False


def main():
    args = get_args()

    # 检查所有必需文件
    required_files = [
        args.tokens, args.encoder, args.decoder, args.joiner, args.keywords_file,
        args.vad_model, args.whisper_encoder, args.whisper_decoder, args.whisper_tokens
    ]
    for f in required_files:
        assert_file_exists(f)

    # 打印 keywords.txt 内容用于调试
    with open(args.keywords_file, 'r', encoding='utf-8') as f:
        print(f"keywords.txt 内容:\n{f.read().strip()}")

    # 初始化组件
    kw_spotter = create_keyword_spotter(args)
    vad = create_vad(args)
    whisper_asr = create_whisper_asr(args)

    # 设置麦克风
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    print(f"使用麦克风: {devices[default_input]['name']}")

    sample_rate = args.sample_rate
    chunk_samples = int(args.chunk_duration * sample_rate)

    # 状态机
    MODE_WAKE = 0
    MODE_ACTIVE = 1
    mode = MODE_WAKE
    kw_stream = kw_spotter.create_stream()

    # 活跃模式使用的变量
    last_valid_speech_time = None  # 最后一次有效语音的时间戳

    print("\n系统已启动，请说出唤醒词...")

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(chunk_samples)
            samples = samples.reshape(-1)

            if mode == MODE_WAKE:
                # ---------- 唤醒词检测 ----------
                kw_stream.accept_waveform(sample_rate, samples)
                while kw_spotter.is_ready(kw_stream):
                    kw_spotter.decode_stream(kw_stream)
                    result = kw_spotter.get_result(kw_stream)
                    if result:
                        print(f"\n检测到唤醒词: {result}")
                        print(f"进入活跃模式（超时 {args.active_timeout} 秒）")
                        mode = MODE_ACTIVE
                        # 重置 VAD
                        vad.reset()
                        last_valid_speech_time = time.time()  # 记录进入活跃模式的时间
                        # 重新创建唤醒词流以便下次唤醒
                        kw_stream = kw_spotter.create_stream()
                        break  # 跳出唤醒词处理循环

            else:  # mode == MODE_ACTIVE
                # 将音频送入 VAD
                vad.accept_waveform(samples)

                # 处理所有完成的语音段
                while not vad.empty():
                    seg = vad.front
                    if seg.samples:
                        speech_audio = np.array(seg.samples, dtype=np.float32)
                        # 只有长度足够的语音段才视为有效交互，重置超时计时器
                        if len(speech_audio) >= 0.5 * sample_rate:
                            success = process_speech_segment(speech_audio, sample_rate, whisper_asr)
                            if success:
                                last_valid_speech_time = time.time()
                        else:
                            print("(忽略过短语音段)")
                    else:
                        print("(语音段无数据，已忽略)")
                    vad.pop()

                # 超时检查：如果超过 active_timeout 没有有效语音，则退出活跃模式
                if last_valid_speech_time is not None:
                    if time.time() - last_valid_speech_time > args.active_timeout:
                        print("\n活跃模式超时（无有效语音），退出到唤醒监听模式。")
                        mode = MODE_WAKE
                        kw_stream = kw_spotter.create_stream()
                        print("已回到唤醒监听模式，请说出唤醒词...")
                        # 重置 VAD 以避免残留数据
                        vad.reset()
                        last_valid_speech_time = None

            # 避免 CPU 占用过高
            time.sleep(0.005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序退出。")