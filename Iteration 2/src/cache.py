import json
import os
from typing import Optional, Dict, Any

from src.models import Answer, Citation


class QueryCache:
    """
    Handles persistent caching of query results to improve performance
    and reduce redundant pipeline executions.
    """

    def __init__(self, cache_file: str = "data/query_cache.json") -> None:
        self.cache_file: str = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """
        Loads the cache from a JSON file.

        If the file does not exist or is corrupted, an empty cache
        is initialized to ensure system robustness.
        """
        if not os.path.exists(self.cache_file):
            self.cache = {}
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                loaded: Any = json.load(f)
                if isinstance(loaded, dict):
                    self.cache = loaded
                else:
                    # Unexpected JSON structure
                    print(f"Warning: Cache file format invalid. Resetting cache.")
                    self.cache = {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load cache from {self.cache_file}. Error: {e}")
            self.cache = {}

    def save(self) -> None:
        """
        Saves the current cache state to disk.

        Automatically creates the target directory if it does not exist.
        """
        try:
            directory: str = os.path.dirname(self.cache_file)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error: Could not save cache to {self.cache_file}. Error: {e}")

    def get(self, question: str) -> Optional[Answer]:
        """
        Retrieves a cached Answer object for a given question.

        Returns:
            Answer if present and valid, otherwise None.
        """
        key: str = question.strip().lower()
        data: Optional[Dict[str, Any]] = self.cache.get(key)

        if data is None:
            return None

        try:
            final_text: str = data["finalText"]
            citations_raw: Any = data.get("citations", [])

            citations: list[Citation] = [
                Citation(**c) for c in citations_raw if isinstance(c, dict)
            ]

            return Answer(finalText=final_text, citations=citations)

        except (KeyError, TypeError) as e:
            print(f"Error: Malformed cache entry for key '{key}'. Error: {e}")
            return None

    def put(self, question: str, answer: Answer) -> None:
        """
        Stores an Answer object in the cache and persists it to disk.
        """
        key: str = question.strip().lower()

        try:
            self.cache[key] = {
                "finalText": answer.finalText,
                "citations": [c.__dict__ for c in answer.citations]
            }
            self.save()
        except AttributeError as e:
            # Defensive programming: Answer object is not well-formed
            print(f"Error: Failed to cache answer for key '{key}'. Error: {e}")
