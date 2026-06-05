"""
file path: src/soundtext4agent/control_plane.py
description: control plane for soundtext4agent

- it's singleton
- it maintains status for soundtext4agent as control plane
- it has status for listen and speak
- the default status is listen
- it can switch status to listen or speak by a status change api
- it has a queue for storing text from listen
- when switched to listen model, it clean up listen queue
- it has a queue for storing text for speak

Missing features (not implemented, kept for future consideration):
- Thread safety for concurrent access (mutex/lock)
- Maximum queue size limit to avoid unbounded growth
- Event/callback notifications on status change (observer pattern)
- Metrics integration (e.g., OpenTelemetry counters for queue lengths, status transitions)
"""
import logging
from collections import deque
from typing import Optional

class ControlPlane:
    """Singleton control plane managing listen/speak status and associated text queues."""
    _instance = None

    def __init__(self):
        # Prevent direct instantiation – use get_instance()
        raise RuntimeError("Use ControlPlane.get_instance() to obtain the singleton instance.")

    @classmethod
    def get_instance(cls) -> 'ControlPlane':
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Internal initialisation of the singleton instance."""
        self.logger = logging.getLogger(__name__)
        self._status = "listen"          # default status
        self.listen_queue = deque()
        self.speak_queue = deque()
        self.logger.info("ControlPlane initialized with status: %s", self._status)

    # ----------------------------------------------------------------------
    # Status property
    # ----------------------------------------------------------------------
    @property
    def status(self) -> str:
        """Current status: 'listen' or 'speak'."""
        return self._status

    # ----------------------------------------------------------------------
    # Status change API
    # ----------------------------------------------------------------------
    def switch_to_listen(self) -> None:
        """Switch the control plane to listen mode. Clears the listen queue."""
        self._status = "listen"
        self.listen_queue.clear()
        self.logger.info("ControlPlane switched to LISTEN mode; listen queue cleared.")

    def switch_to_speak(self) -> None:
        """Switch the control plane to speak mode."""
        self._status = "speak"
        self.logger.info("ControlPlane switched to SPEAK mode.")

    # ----------------------------------------------------------------------
    # Listen queue operations
    # ----------------------------------------------------------------------
    def enqueue_listen_text(self, text: str) -> None:
        """Add text to the listen queue (typically while in listen mode)."""
        self.listen_queue.append(text)
        self.logger.debug("Enqueued listen text: %s", text)

    def dequeue_listen_text(self) -> Optional[str]:
        """Pop the oldest text from the listen queue. Returns None if empty."""
        if self.listen_queue:
            text = self.listen_queue.popleft()
            self.logger.debug("Dequeued listen text: %s", text)
            return text
        return None

    # ----------------------------------------------------------------------
    # Speak queue operations
    # ----------------------------------------------------------------------
    def enqueue_speak_text(self, text: str) -> None:
        """Add text to the speak queue for later speech output."""
        self.speak_queue.append(text)
        self.logger.debug("Enqueued speak text: %s", text)

    def dequeue_speak_text(self) -> Optional[str]:
        """Pop the oldest text from the speak queue. Returns None if empty."""
        if self.speak_queue:
            text = self.speak_queue.popleft()
            self.logger.debug("Dequeued speak text: %s", text)
            return text
        return None

    # ----------------------------------------------------------------------
    # Manual queue clearing (supplementary)
    # ----------------------------------------------------------------------
    def clear_listen_queue(self) -> None:
        """Manually clear the listen queue."""
        self.listen_queue.clear()
        self.logger.debug("Listen queue manually cleared.")

    def clear_speak_queue(self) -> None:
        """Manually clear the speak queue."""
        self.speak_queue.clear()
        self.logger.debug("Speak queue manually cleared.")


"""
Example usage:
    # (In another module)
    from soundtext4agent.control_plane import ControlPlane

    # Obtain the singleton instance
    cp = ControlPlane.get_instance()

    # Default status is 'listen'
    assert cp.status == "listen"

    # Push some listen text
    cp.enqueue_listen_text("Hello from listen mode")
    print(cp.dequeue_listen_text())   # "Hello from listen mode"

    # Switch to speak mode
    cp.switch_to_speak()
    assert cp.status == "speak"

    # Queue text for speech
    cp.enqueue_speak_text("I will speak this")
    print(cp.dequeue_speak_text())    # "I will speak this"

    # Switching to listen clears the listen queue
    cp.enqueue_listen_text("temporary")
    cp.switch_to_listen()
    print(cp.dequeue_listen_text())   # None (queue was cleared)
"""