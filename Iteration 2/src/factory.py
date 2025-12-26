import json
import os
from typing import Dict, Any, List

from src.models import Chunk, IndexEntry, KeywordIndex
from src.pipeline import RagOrchestrator
from src.cache import QueryCache

from src.impl import (
    ConfigurableIntentDetector,
    HeuristicQueryWriter,
    KeywordRetriever,
    SimpleReranker,
    CosineReranker,
    KeywordAnswerAgent,
    VectorAnswerAgent,
)


class PipelineFactory:
    """
    Factory class responsible for constructing the RAG pipeline
    based on configuration settings.
    """

    DEFAULT_INTENT_RULES: Dict[str, List[str]] = {
        "STAFF_LOOKUP": ["hoca", "ofis", "mail", "iletişim", "kimdir", "başkan", "odası", "yeri"],
        "COURSE_INFO": ["ders", "kredi", "ects", "akts", "önkoşul", "dönem"],
        "POLICY_FAQ": ["yönetmelik", "yönerge", "sınav", "staj", "mezuniyet", "çap", "yatay"],
        "REGISTRATION": ["kayıt", "dondurma", "harç"],
    }

    @staticmethod
    def create(config: Dict[str, Any]) -> RagOrchestrator:
        """
        Creates and wires all pipeline components.
        """
        chunks: List[Chunk] = PipelineFactory._load_chunks()
        index: KeywordIndex = PipelineFactory._load_index()

        # Persistent cache for query results
        query_cache: QueryCache = QueryCache()

        intent_rules = config.get("pipeline", {}).get(
            "intent_rules", PipelineFactory.DEFAULT_INTENT_RULES
        )
        intent_detector = ConfigurableIntentDetector(intent_rules)

        reranker_config: Dict[str, Any] = config.get("pipeline", {}).get("reranker", {})
        reranker_type: str = reranker_config.get("type", "simple").lower()

        if reranker_type == "cosine":
            reranker = CosineReranker(chunks)
            answer_agent = VectorAnswerAgent()
        else:
            reranker = SimpleReranker(chunks)
            answer_agent = KeywordAnswerAgent()

        query_writer = HeuristicQueryWriter()
        retriever = KeywordRetriever()

        return RagOrchestrator(
            intent_detector,
            query_writer,
            retriever,
            reranker,
            answer_agent,
            index,
            query_cache,
        )

    @staticmethod
    def _load_chunks() -> List[Chunk]:
        """
        Loads document chunks from disk.
        """
        path: str = "data/chunks.json"
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
                if not isinstance(data, list):
                    return []
                return [Chunk(**c) for c in data if isinstance(c, dict)]
        except (json.JSONDecodeError, IOError, TypeError):
            return []

    @staticmethod
    def _load_index() -> KeywordIndex:
        """
        Loads keyword index structure from disk.
        """
        path: str = "data/index.json"
        if not os.path.exists(path):
            return KeywordIndex({})

        try:
            with open(path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
                source_data: Any = data.get("indexMap", data)

                if not isinstance(source_data, dict):
                    return KeywordIndex({})

                index_map: Dict[str, List[IndexEntry]] = {}
                for term, entries in source_data.items():
                    if not isinstance(entries, list):
                        continue
                    index_map[term] = [
                        IndexEntry(**e) for e in entries if isinstance(e, dict)
                    ]

                return KeywordIndex(index_map)

        except (json.JSONDecodeError, IOError, TypeError):
            return KeywordIndex({})
