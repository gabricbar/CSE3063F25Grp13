from abc import ABC, abstractmethod
from typing import List
from .models import Intent, Hit, KeywordIndex, Answer

class IntentDetector(ABC):
    @abstractmethod
    def detect(self, question: str) -> Intent:
        pass

class QueryWriter(ABC):
    @abstractmethod
    def write(self, question: str, intent: Intent) -> List[str]:
        pass

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query_terms: List[str], index: KeywordIndex) -> List[Hit]:
        pass

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query_terms: List[str], hits: List[Hit]) -> List[Hit]:
        pass

class AnswerAgent(ABC):
    @abstractmethod
    def answer(self, question: str, top_hits: List[Hit]) -> Answer:
        pass
