import time
import json
import os
from dataclasses import dataclass
from typing import List, Callable

# Equivalent of Java's TraceEvent class
@dataclass
class TraceEvent:
    stage: str
    inputs: str
    outputsSummary: str
    timingMs: float
    errors: str = None
    timestamp: int = 0

    def __post_init__(self):
        # Add timestamp (in ms) when the event is created
        self.timestamp = int(time.time() * 1000)


# Equivalent of Java's JsonlTraceSink class
class JsonlTraceSink:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        
        # Auto-create directory if it does not exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def accept(self, event: TraceEvent):
        # Create JSON structure identical to Java implementation
        data = {
            "stage": event.stage,
            # Clean newline characters to keep log entries single-line (JSONL)
            "inputs": event.inputs.replace('"', "'").replace("\n", " ") if event.inputs else "",
            "outputsSummary": event.outputsSummary.replace('"', "'").replace("\n", " ") if event.outputsSummary else "",
            "timingMs": event.timingMs
        }
        
        if event.errors:
            data["errors"] = event.errors.replace('"', "'")

        # Append JSON object to file (JSONL format)
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")  # new line for JSONL
        except Exception as e:
            print(f"Logging error: {e}")


# Equivalent of Java's TraceBus class (Observer Pattern)
class TraceBus:
    _listeners: List[Callable[[TraceEvent], None]] = []

    @staticmethod
    def register(listener):
        """Register a new listener (sink)."""
        # If listener is an object with an 'accept' method, register that method
        if hasattr(listener, 'accept'):
            TraceBus._listeners.append(listener.accept)
        else:
            # Otherwise, listener itself is a callable
            TraceBus._listeners.append(listener)

    @staticmethod
    def push(stage: str, message: str):
        """Simple logging method for backward compatibility."""
        TraceBus.push_full(stage, "", message, 0, None)

    @staticmethod
    def push_full(stage: str, inputs: str, outputs_summary: str, timing_ms: int, errors: str = None):
        """Push a detailed log event to all listeners."""
        event = TraceEvent(stage, inputs, outputs_summary, timing_ms, errors)
        
        # Notify all registered listeners
        for listener in TraceBus._listeners:
            listener(event)
