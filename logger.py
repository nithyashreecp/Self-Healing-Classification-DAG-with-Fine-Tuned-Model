# logger.py
import logging
import json
from datetime import datetime

class JSONLogger:
    """
    Simple structured JSON logger that appends JSON lines to a file.
    Each log entry includes timestamp, event, and details dict.
    """

    def __init__(self, filename: str = "demo_logs.log"):
        self.filename = filename

    def _write(self, obj: dict):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    def log(self, event: str, details: dict):
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "details": details,
        }
        self._write(payload)

