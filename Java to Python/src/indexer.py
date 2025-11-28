import os
import json
from src.models import Chunk, IndexEntry, KeywordIndex

class IndexerMain:

    @staticmethod
    def tokenize(text: str):
        import re
        text = text.lower()
        tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text)
        return tokens

    @staticmethod
    def process_file(path, filename, all_chunks, raw_index_map):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        doc_id = filename.replace(".txt", "")
        chunk_id = 0

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            tokens = IndexerMain.tokenize(trimmed)
            if not tokens:
                continue

            start_offset = 0
            end_offset = len(trimmed)

            # ----- CREATE ADJUSTED CHUNK -----
            chunk = Chunk(
                docId=doc_id,
                chunkId=chunk_id,
                rawText=trimmed,
                startOffset=start_offset,
                endOffset=end_offset
            )
            all_chunks.append(chunk)

            # ----- ADD INDEX -----
            for t in tokens:
                if t not in raw_index_map:
                    raw_index_map[t] = []

                raw_index_map[t].append(
                    IndexEntry(
                        docId=doc_id,
                        chunkId=chunk_id,
                        tf=1
                    )
                )

            chunk_id += 1

    @staticmethod
    def main():
        corpus_dir = "data/corpus"
        all_chunks = []
        raw_index_map = {}

        print("\n=== PYTHON INDEXER STARTING ===")

        for filename in os.listdir(corpus_dir):
            if filename.endswith(".txt"):
                print(f"Processing: {filename}")
                IndexerMain.process_file(
                    os.path.join(corpus_dir, filename),
                    filename,
                    all_chunks,
                    raw_index_map
                )

        print(f"=== DONE. Number of Chunks: {len(all_chunks)} ===")

        chunks_data = [c.__dict__ for c in all_chunks]

        with open("data/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        index_map = {}
        for term, entries in raw_index_map.items():
            entry_dicts = [e.__dict__ for e in entries]
            index_map[term] = entry_dicts

        with open("data/index.json", "w", encoding="utf-8") as f:
            json.dump({"indexMap": index_map}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    IndexerMain.main()
