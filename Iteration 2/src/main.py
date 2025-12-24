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
    CosineReranker,
    TemplateAnswerAgent
)
from src.tracing import TraceBus, JsonlTraceSink


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
# BATCH MODE
# -------------------------------
def run_batch(orchestrator: RagOrchestrator, batch_path: str, out_path: str):
    """Read questions from JSONL and write answers to JSONL."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(batch_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON on line {line_no} in {batch_path}")

            qid = item.get("id", str(line_no))
            question = item.get("question") or item.get("q")
            if not question:
                raise SystemExit(f"Missing 'question' field on line {line_no} in {batch_path}")

            t0 = time.perf_counter()
            answer_obj = orchestrator.run(question)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            row = {
                "id": qid,
                "question": question,
                "answer": answer_obj.finalText,
                "citations": [
                    {
                        "docId": c.docId,
                        "sectionId": c.sectionId,
                        "startOffset": c.startOffset,
                        "endOffset": c.endOffset,
                    }
                    for c in (answer_obj.citations or [])
                ],
                "latency_ms": latency_ms,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="MiniRAG Python (Iteration 2)")
    parser.add_argument("--config", help="Config file (optional for this iteration)")

    # New argument for selecting reranker
    parser.add_argument("--reranker", choices=["simple", "cosine"], default="simple", 
                        help="Select the reranking strategy (default: simple)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--q", help="Single question (interactive mode)")
    group.add_argument("--batch", help="Path to questions.jsonl for batch mode")

    parser.add_argument("--out", help="Path to answers.jsonl (required with --batch)")

    args = parser.parse_args()

    if args.batch and not args.out:
        raise SystemExit("--out is required when using --batch")

    # 1. Logging
    setup_tracing()

    # 2. Data
    chunks, index = load_data()

    # 3. Component Selection
    if args.reranker == "cosine":
        reranker = CosineReranker(chunks)
        print("Using CosineReranker")
    else:
        reranker = SimpleReranker(chunks)
        print("Using SimpleReranker")

    # 4. Pipeline
    orchestrator = RagOrchestrator(
        RuleBasedIntentDetector(),
        HeuristicQueryWriter(),
        KeywordRetriever(),
        reranker,
        TemplateAnswerAgent(),
        index
    )

    # 5. Execute
    if args.batch:
        run_batch(orchestrator, args.batch, args.out)
        print(f"Wrote batch answers to {args.out}")
    else:
        answer = orchestrator.run(args.q)
        print(str(answer))


if __name__ == "__main__":
    main()