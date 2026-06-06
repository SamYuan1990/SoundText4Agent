"""
example.py – Echo demonstration using STT + TTS.

Prerequisites:
- A working SoundToTextManager (from soundtext4agent.SoundTo.stt).
- A working TextToManager (from soundtext4agent.TextTo.ttmgr).
- Valid STT and TTS model files (update paths below).
- soundtext4agent.control_plane for queue management.
"""

import time
import logging
from soundtext4agent.SoundTo.stt import SoundToTextManager
from soundtext4agent.TextTo.ttmgr import TextToManager
from soundtext4agent.control_plane import ControlPlane

# ----------------------------------------------------------------------
# Setup logging (info + debug)
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("echo_demo")

def echo_loop(duration_seconds: int = 60):
    """
    Run the echo demonstration for a given duration (seconds).
    - Listens for wake word then voice commands.
    - For each recognised phrase, plays it back via TTS.
    """
    # 1. Initialize TTS manager
    logger.info("Initialising TextToManager...")
    tts_manager = TextToManager.configure()
    tts_manager.start()
    logger.info("TTS manager started.")

    # 2. Initialize STT manager
    logger.info("Initialising SoundToTextManager...")
    stt_manager = SoundToTextManager()
    stt_manager.start()
    logger.info("STT manager started – say the wake word then speak.")

    # 3. Control plane (used to bridge STT and TTS if needed, but we can also
    #    directly call tts_manager.enqueue if such method exists.
    #    Here we use the control plane's queue for flexibility.
    cp = ControlPlane.get_instance()

    # 4. Main loop: collect STT results and enqueue them for TTS playback
    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            try:
                # Wait for a recognised text from STT (timeout 1 second for responsiveness)
                #result = stt_manager.results.get(timeout=1.0)
                #recognised_text = result.get("text", "").strip()
                recognised_text = cp.dequeue_listen_text()
                if recognised_text:
                    logger.info(f"Recognised: {recognised_text}")
                    # Echo: send text to TTS via control plane
                    cp.enqueue_speak_text(recognised_text)
                else:
                    logger.debug("Empty recognition result.")
            except Exception as e:
                # Timeout is normal, ignore
                if "Empty" not in str(type(e).__name__):
                    logger.debug(f"Queue get interrupted: {e}")
                continue
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # 5. Shutdown gracefully
        logger.info("Stopping STT and TTS managers...")
        stt_manager.stop()
        tts_manager.stop()
        logger.info("Echo demonstration finished.")


if __name__ == "__main__":
    # Run for 60 seconds (or until Ctrl+C)
    echo_loop(duration_seconds=60)