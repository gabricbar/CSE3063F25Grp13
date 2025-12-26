from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

class Intent(Enum):
    REGISTRATION = "REGISTRATION"
    STAFF_LOOKUP = "STAFF_LOOKUP"
    POLICY_FAQ = "POLICY_FAQ"
    COURSE_INFO = "COURSE_INFO"
    UNKNOWN = "UNKNOWN"

@dataclass
class Chunk:
    docId: str
    chunkId: int
    rawText: str
    startOffset: int
    endOffset: int
    sectionId: Optional[str] = None
    embedding: Optional[List[float]] = None  # adding for iteration 2

@dataclass
class IndexEntry:
    docId: str
    chunkId: int
    tf: int

@dataclass
class KeywordIndex:
    # Token -> List of IndexEntry
    indexMap: Dict[str, List[IndexEntry]] = field(default_factory=dict)

@dataclass(order=True)
class Hit:
    # sort_index field is used for comparison to emulate Java's compareTo
    # Java logic: Score DESC, DocID ASC, ChunkID ASC
    # Python default is ASC, so we negate score for sorting.
    sort_index: tuple = field(init=False, repr=False)
    
    docId: str
    chunkId: int
    score: float
    chunkText: Optional[str] = None
    embedding: Optional[List[float]] = None # adding for iteration 2

    def __post_init__(self):
        # Score negated for DESC sort, others ASC
        self.sort_index = (-self.score, self.docId, self.chunkId)

@dataclass
class Citation:
    docId: str
    sectionId: str
    startOffset: int
    endOffset: int

    def __str__(self):
        sec = self.sectionId if self.sectionId else "General"
        return f"{self.docId}:{sec}:{self.startOffset}-{self.endOffset}"

@dataclass
class Answer:
    finalText: str
    citations: List[Citation]

    def __str__(self):
        cit_str = " ".join([f"[{str(c)}]" for c in self.citations])
        return f"{self.finalText}\nSources: {cit_str}"