"""Current target description from the upstream node, versioned for the client."""
import threading

MAX_QUERY_CHARS = 300


class QueryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._text = None
        self._version = 0

    def get(self):
        with self._lock:
            return self._text, self._version

    def set(self, text):
        text = str(text).strip()
        if not text or len(text) > MAX_QUERY_CHARS:
            raise ValueError(f'query must be 1-{MAX_QUERY_CHARS} characters')
        with self._lock:
            if text != self._text:
                self._text = text
                self._version += 1
            return self._version

    def clear(self):
        with self._lock:
            if self._text is not None:
                self._text = None
                self._version += 1
            return self._version
