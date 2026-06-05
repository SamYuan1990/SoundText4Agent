"""
File path: src/soundtext4agent/TextTo/ttmgr.py
Description: Text-to-any manager for the TextTo package.

Features and design goals:
- Singleton manager that owns a TTS (TextToSpeech) and a TTL (TextToLip) engine.
- Runs a background thread that continuously fetches text from the ControlPlane's
  speak queue (using `ControlPlane.get_instance().dequeue_speak_text()`).
- For each text, processes TTS and TTL in parallel and returns both the generated
  audio file path and the lip-sync events.
- Provides a results queue to deliver processed outcomes to other components.
- Supports graceful start/stop of the background thread.
- Configurable TTS engine via a dedicated `configure()` call.
- Uses logging for info and debug level messages.
"""

import logging
import threading
import queue
import uuid
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from soundtext4agent.TextTo.tts import TextToSpeech
from soundtext4agent.TextTo.ttl import TextToLip
from soundtext4agent.control_plane import ControlPlane

logger = logging.getLogger(__name__)


class TextToManager:
    """
    Manages speech and lip-sync generation for text from the control plane.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # Engine references (initialised during configure())
        self.tts = None
        self.ttl = None
        # Threading control
        self._thread = None
        self._stop_event = threading.Event()
        # Queue to hold processing results (wav_path, lip_events, text)
        self.results = queue.Queue()
        # Configuration flag
        self._configured = False

    @classmethod
    def configure(cls, tts_config: dict):
        """
        Initialise the TTS engine with the given configuration and obtain the TTL
        instance. Must be called before starting the manager.

        Returns the singleton instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            inst = cls._instance
        # It is safe to configure only once.
        if not inst._configured:
            inst.tts = TextToSpeech.get_instance(tts_config)
            inst.ttl = TextToLip.instance()
            inst._configured = True
            logger.info("TextToManager configured successfully.")
        return inst

    @classmethod
    def get_instance(cls):
        """
        Return the singleton instance. Must have been configured previously.
        """
        if cls._instance is None or not cls._instance._configured:
            raise RuntimeError("TextToManager not configured. Call configure() first.")
        return cls._instance

    def start(self):
        """Launch the background processing thread."""
        if not self._configured:
            raise RuntimeError("Cannot start before configure().")
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Manager thread is already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="TextToManagerThread", daemon=True)
        self._thread.start()
        logger.info("TextToManager thread started.")

    def stop(self):
        """Signal the background thread to stop and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("Manager thread did not terminate in time.")
        else:
            logger.info("TextToManager thread stopped.")

    def _run(self):
        """
        Main loop: dequeue text from the ControlPlane, process it, and push results
        to the internal results queue.
        """
        cp = ControlPlane.get_instance()
        while not self._stop_event.is_set():
            try:
                # Blocking call – the control plane should provide a way to wake up on stop.
                # If the underlying call does not support interruption, consider a timeout
                # wrapper. For now we assume dequeue_speak_text returns immediately when
                # the stop event is set (e.g., the control plane checks a flag).
                print("try to get msg from queue")
                text = cp.dequeue_speak_text()
                if text is None:
                    # A None may be used as a sentinel to stop the loop.
                    time.sleep(5)
                    continue
                print(text)
                logger.debug(f"Dequeued text for processing: {text[:50]}...")
                # Process in parallel
                wav_path, lip_events = self._process_text(text)
                # Push result (the caller can retrieve it from self.results)
                self.results.put({
                    "text": text,
                    "wav_path": wav_path,
                    "lip_events": lip_events,
                })
                logger.info(f"Processed text successfully, saved to {wav_path}")
            except Exception as e:
                logger.error(f"Error processing text: {e}", exc_info=True)

    def _process_text(self, text: str):
        """
        Process a single text string: generate speech audio and lip-sync events
        in parallel using a thread pool.

        Returns:
            tuple: (wav_file_path, lip_events) where lip_events is a list of
                   (phoneme, start_time, end_time).
        """
        # Use a small thread pool for the two independent tasks.
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit TTS generation (saving to a temp file)
            tts_future = executor.submit(self._generate_audio, text)
            # Submit TTL conversion
            ttl_future = executor.submit(self.ttl.convert, text, "en-us")

            # Wait for both to finish (order doesn't matter)
            wav_path = tts_future.result()
            lip_events = ttl_future.result()
        return wav_path, lip_events

    def _generate_audio(self, text: str) -> str:
        """
        Generate audio for the given text and save it to a uniquely named WAV file.

        Returns:
            str: the path to the created WAV file.
        """
        # Create a unique filename in the current working directory (or a configurable dir)
        self.tts.speak(text)
        filename = f"speech_{uuid.uuid4().hex[:12]}.wav"
        # Use the TTS engine's built-in save method
        self.tts.save_wav(filename, text, sid=0, speed=1.0)
        return os.path.abspath(filename)


# =============================================================================
# Example usage (to be used as a reference by other modules)
# =============================================================================
"""
# In another module (e.g., the main application):
from soundtext4agent.TextTo.ttmgr import TextToManager

# 1. Configure the manager (also initialises the underlying TTS/TTL engines)
tts_config = {
    "engine": "matcha",
    "model_path": "/path/to/model.onnx",
    "vocoder": "/path/to/vocoder.onnx",
    "lexicon": "/path/to/lexicon.txt",
    "tokens": "/path/to/tokens.txt",
    "data_dir": "/path/to/espeak-ng-data",
    "num_threads": 1,
    "play_audio": False,   # Disable automatic playback inside the manager
}
manager = TextToManager.configure(tts_config)

# 2. Start the background processing thread
manager.start()

# ... later, text is enqueued into the control plane by some other logic ...
# The manager thread will automatically dequeue and process it.

# 3. Retrieve results (blocking or non-blocking) as needed
try:
    result = manager.results.get(timeout=10)   # block until a result is available
    print(f"Text processed: {result['text']}")
    print(f"WAV saved at: {result['wav_path']}")
    print(f"Lip events: {result['lip_events']}")
except queue.Empty:
    print("No result yet.")

# 4. Stop the manager when shutting down
manager.stop()
"""