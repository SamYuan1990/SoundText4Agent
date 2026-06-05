"""
example.py – Real‑world usage of TextToManager.

Prerequisites:
- A working ControlPlane (soundtext4agent.control_plane) that supports
  enqueue_speak_text() and dequeue_speak_text().
- Valid TTS model files (update tts_config below).
- soundtext4agent.TextTo.tts and ttl installed.
"""

import time
import threading
import logging
from soundtext4agent.TextTo.ttmgr import TextToManager
from soundtext4agent.control_plane import ControlPlane

# ----------------------------------------------------------------------
# Setup logging (info + debug)
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("example")

# ----------------------------------------------------------------------
# Helper: enqueue text into the real ControlPlane
# In your application this would be done by whatever generates speak commands.
# ----------------------------------------------------------------------
def feed_text_into_plane(texts):
    cp = ControlPlane.get_instance()
    # Assuming the control plane exposes an enqueue method; if not,
    # this is where you would call the appropriate API.
    for t in texts:
        cp.enqueue_speak_text(t)
        logger.info(f"Enqueued text: {t[:30]}...")
        time.sleep(0.1)  # give the manager thread a chance to process

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # 1. Configure the TextToManager with your TTS settings.
    tts_config = {
        "engine": "matcha",
        "model_path": "/Users/yuanyi/OpenSource/TTSlocal/matcha-icefall-zh-en/model-steps-3.onnx",  # 请修改为实际路径
        "vocoder": "/Users/yuanyi/OpenSource/TTSlocal/vocos-16khz-univ.onnx",
        "lexicon": "/Users/yuanyi/OpenSource/TTSlocal/matcha-icefall-zh-en/lexicon.txt",
        "tokens": "/Users/yuanyi/OpenSource/TTSlocal/matcha-icefall-zh-en/tokens.txt",            # 请修改
        "data_dir": "/Users/yuanyi/OpenSource/TTSlocal/matcha-icefall-zh-en/espeak-ng-data",      # 请修改
        "num_threads": 1,
        "play_audio": True,  # we will handle playback ourselves if needed
    }

    logger.info("Configuring TextToManager...")
    manager = TextToManager.configure(tts_config)

    # 2. Start the background processing thread.
    manager.start()
    logger.info("Manager started – ready to process texts from the ControlPlane.")

    # 3. Simulate text being fed from another part of the system.
    #    In a real app this would happen asynchronously.
    test_texts = [
        "Hello, welcome to the speech demonstration.",
        "This is a second phrase, processed in parallel.",
        "One final sentence to show the workflow.",
    ]

    # Use a separate thread so the main thread can monitor results.
    feeder = threading.Thread(target=feed_text_into_plane, args=(test_texts,), daemon=True)
    feeder.start()

    # 4. Retrieve results as they become available.
    #    The manager puts each processed dict into manager.results.
    expected = len(test_texts)
    for i in range(expected):
        try:
            result = manager.results.get(timeout=15.0)  # adjust timeout as needed
            logger.info(
                f"Result {i+1}: text='{result['text'][:40]}...', "
                f"wav_path='{result['wav_path']}', "
                f"lip_events_count={len(result['lip_events'])}"
            )
            print(result['lip_events'])
        except Exception as e:
            logger.error(f"Timeout or error waiting for result {i+1}: {e}")
            break

    # 5. Shutdown the manager gracefully.
    logger.info("Stopping TextToManager...")
    manager.stop()
    logger.info("Manager stopped.")

    # 6. (Optional) Verify the audio files exist.
    import os
    while not manager.results.empty():
        remaining = manager.results.get()
        if os.path.exists(remaining['wav_path']):
            logger.info(f"Audio file exists: {remaining['wav_path']}")

    logger.info("Example finished.")


if __name__ == "__main__":
    main()