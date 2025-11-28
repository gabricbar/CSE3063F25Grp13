import argparse
import json
import os
from datetime import datetime
import time

from src.pipeline import RagOrchestrator
from src.models import Chunk, IndexEntry, KeywordIndex
from src.impl import (
    RuleBasedIntentDetector,
    HeuristicQueryWriter,
    KeywordRetriever,
    SimpleReranker,
    TemplateAnswerAgent
)
from src.tracing import TraceBus, JsonlTraceSink
from src.core import (
    IntentDetector,
    QueryWriter,
    Retriever,
    Reranker,
    AnswerAgent
)

# -------------------------------
# DATA LOAD
# -------------------------------
def load_data():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks_raw = json.load(f)

    all_chunks = [
        Chunk(
            docId=c["docId"],
            chunkId=c["chunkId"],
            rawText=c["rawText"],
            startOffset=c.get("startOffset", 0),
            endOffset=c.get("endOffset", 0),
            sectionId=c.get("sectionId")
        )
        for c in chunks_raw
    ]

    with open("data/index.json", "r", encoding="utf-8") as f:
        index_raw = json.load(f)

    index_map = {}
    for term, entries in index_raw["indexMap"].items():
        index_map[term] = [
            IndexEntry(
                docId=e["docId"],
                chunkId=e["chunkId"],
                tf=e["tf"]
            )
            for e in entries
        ]

    return all_chunks, KeywordIndex(indexMap=index_map)


# -------------------------------
# TRACE SETUP
# -------------------------------
def setup_tracing():
    if not os.path.exists("logs"):
        os.mkdir("logs")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # logs/run-timestamp.jsonl
    TraceBus.register(JsonlTraceSink(f"logs/run-{timestamp}.jsonl"))

    # Global file
    TraceBus.register(JsonlTraceSink("rag_trace.jsonl"))


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="MiniRAG Python")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--q", help="Question", required=True)
    args = parser.parse_args()

  
    # 1. Logging
    setup_tracing()

    # 2. Data
    chunks, index = load_data()

    # 3. Pipeline
    orchestrator = RagOrchestrator(
        RuleBasedIntentDetector(),
        HeuristicQueryWriter(),
        KeywordRetriever(),
        SimpleReranker(chunks),
        TemplateAnswerAgent(),
        index
    )

    # 4. Execute (only single question)
    answer = orchestrator.run(args.q)
    print(answer)


if __name__ == "__main__":
    main()
