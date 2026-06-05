"""
file path: src/soundtext4agent/TextTo/ttl.py
description: text to lip shape token service

Features and design goals
- Singleton: only one instance of TextToLip is created and reused.
- Receives text and converts it to lip shape tokens (phonemes with start/end time) based on espeak-ng and piper_phonemize.
- Provides a timeline of (phoneme, start_sec, end_sec) for lip sync.
- Logging at info and debug levels for monitoring and troubleshooting.
- Graceful fallback when espeak-ng fails (heuristic timeline generation).
"""

import os
import re
import sys
import logging
import tempfile
import subprocess
from piper_phonemize import phonemize_espeak
# "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"

# ---------- module-level logger ----------
logger = logging.getLogger(__name__)


# ---------- helper functions (unchanged, except added logging) ----------
def parse_pho_lines(lines, frame_ms=10):
    """Parse espeak-ng output lines: phoneme duration_frames"""
    timings = []
    current_time = 0.0
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("lang:", "name:", "phonemes:", "frames:", "voice:")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        phoneme = parts[0]
        dur_frames = int(parts[1])
        dur_sec = dur_frames * frame_ms / 1000.0
        timings.append((phoneme, current_time, current_time + dur_sec))
        current_time += dur_sec
    logger.debug(f"Parsed {len(timings)} phoneme timings")
    return timings


def get_phoneme_timings(text: str, voice: str = "en-us", frame_ms: int = 10):
    """
    Invoke espeak-ng to get phoneme timing information.
    Tries -x -w to produce .pho file first; falls back to -m stdout.
    Returns list of (phoneme, start_sec, end_sec).

    Now with detailed error logging so failures are not swallowed.
    """
    logger.info(f"Getting phoneme timings for text (first 30 chars): {text[:30]!r} with voice={voice}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "out.wav")
        pho_path = os.path.splitext(wav_path)[0] + ".pho"

        # Strategy 1: -x -w auto-generate .pho
        cmd1 = [
            "espeak-ng", "-v", voice,
            "-x", "-q",
            "-w", wav_path,
            text
        ]
        try:
            subprocess.run(cmd1, check=True, capture_output=True)
            if os.path.isfile(pho_path):
                with open(pho_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                return parse_pho_lines(lines, frame_ms)
        except FileNotFoundError:
            print("espeak-ng command not found. Please install espeak-ng and ensure it is on PATH.")
            return []
        except subprocess.CalledProcessError as e:
            stderr_detail = e.stderr.decode(errors="replace") if e.stderr else ""
            stdout_detail = e.stdout.decode(errors="replace") if e.stdout else ""
            print(
                f"Strategy 1 failed (returncode={e.returncode}). "
                f"stderr: {stderr_detail.strip()}, stdout: {stdout_detail.strip()}"
            )

        # Strategy 2: -m stdout
        #cmd2 = ["espeak-ng", "-v", voice, "-m", text]
        #try:
        #    result = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        #    lines = result.stdout.strip().split('\n')
        #    return parse_pho_lines(lines, frame_ms)
        #except FileNotFoundError:
        #    print("espeak-ng command not found.")
        #    return []
        #except subprocess.CalledProcessError as e:
        #    stderr_detail = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace") if e.stderr else ""
        #    stdout_detail = e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace") if e.stdout else ""
        #    print(
        #        f"Strategy 2 failed (returncode={e.returncode}). "
        #        f"stderr: {stderr_detail.strip()}, stdout: {stdout_detail.strip()}"
        #    )
        #    return []

    return []


def generate_phoneme_timeline(text: str):
    """
    Fallback: generate a rough timeline using regex tokenisation and fixed 80ms per phoneme.
    Works for mixed Chinese / English text without requiring espeak-ng.
    """
    logger.info("Using fallback heuristic phoneme timeline")
    # Regex to split into Chinese characters and English words/punctuation
    pattern = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9 ,.!\?]+")
    all_events = []
    offset = 0.0

    for match in pattern.finditer(text):
        segment = match.group().strip()
        if not segment:
            continue

        if re.match(r"[\u4e00-\u9fff]+", segment):
            # 中文：每个字作为一个音素
            #phonemes = list(segment)
            tokens_list = phonemize_espeak(segment, "cmn")
            phonemes = sum(tokens_list, [])
        else:
            # 英文：使用 piper_phonemize
            tokens_list = phonemize_espeak(segment, "en-us")
            phonemes = sum(tokens_list, [])

        if not phonemes:
            continue

        # 简单时长分配：每个音素固定 80ms
        dur_per_ph = 0.080  # 秒
        t = 0.0
        for ph in phonemes:
            all_events.append((ph, offset + t, offset + t + dur_per_ph))
            t += dur_per_ph
        offset += t

    return all_events


# ---------- Singleton Class ----------
class TextToLip:
    """
    Singleton service that converts text to lip shape tokens (phonemes with time stamps).
    Use TextToLip.instance() to get the shared object.
    """
    _instance = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("TextToLip singleton initialised")

    @classmethod
    def instance(cls):
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def convert(self, text: str, voice: str = "en-us"):
        """
        Convert input text to a list of (phoneme, start_sec, end_sec) tuples.
        Uses espeak-ng with fallback to heuristic timeline.

        Args:
            text: string to be converted.
            voice: espeak-ng voice name (default "en-us").

        Returns:
            List of (phoneme, start_sec, end_sec)
        """
        self.logger.info(f"Converting text to lip tokens (voice={voice})")
        self.logger.debug(f"Text length: {len(text)} characters")

        # Try espeak-ng first
        timings = get_phoneme_timings(text, voice)
        if timings:
            self.logger.info(f"Successfully obtained {len(timings)} phoneme events from espeak-ng")
            return timings

        # Fallback to heuristic
        self.logger.warning("espeak-ng failed, falling back to heuristic timeline")
        return generate_phoneme_timeline(text)


# ---------- Convenience function ----------
def text_to_lip_tokens(text: str, voice: str = "en-us"):
    """Convenience function that calls the singleton."""
    return TextToLip.instance().convert(text, voice)


# ========== Example usage ==========
# To use this module from another file:
#
# from soundtext4agent.TextTo.ttl import TextToLip, text_to_lip_tokens
#
# # Obtain the singleton and convert text
# ttl = TextToLip.instance()
# lip_events = ttl.convert("Hello world 你好", voice="en-us")
# for phoneme, start, end in lip_events:
#     print(f"{phoneme}: {start:.3f}s - {end:.3f}s")
#
# # Or use the convenience function directly
# lip_events = text_to_lip_tokens("Hi there")