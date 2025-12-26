import os
import json
import re
from typing import List, Dict, Set, Any
from src.models import Chunk, IndexEntry
from src.utils import get_embedding

class IndexerMain:
    """
    Handles document processing, semantic chunking, and index generation.
    Now includes an auto-purge feature for the query cache.
    """
    MAX_CHUNK_CHARS: int = 1000
    OVERLAP_CHARS: int = 150
    CACHE_FILE: str = "data/query_cache.json" # Path to the cache file 

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Cleans and tokenizes text into alphanumeric words."""
        text = text.lower()
        return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text)

    @staticmethod
    def enforce_max_length(chunks: List[str]) -> List[str]:
        """
        Ensures chunks do not exceed MAX_CHUNK_CHARS using a sliding window.
        Prevents word splitting by looking for the last space.
        """
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= IndexerMain.MAX_CHUNK_CHARS:
                final_chunks.append(chunk)
            else:
                start = 0
                while start < len(chunk):
                    end = min(start + IndexerMain.MAX_CHUNK_CHARS, len(chunk))
                    if end < len(chunk):
                        safe_end = chunk.rfind(' ', start, end)
                        if safe_end != -1 and safe_end > start:
                            end = safe_end
                    
                    sub_chunk = chunk[start:end].strip()
                    if len(sub_chunk) > 20:
                        final_chunks.append(sub_chunk)
                    
                    start = end - IndexerMain.OVERLAP_CHARS
                    if start < 0: start = 0
                    if end == len(chunk): break
        return final_chunks

    @staticmethod
    def split_academic_staff(text: str) -> List[str]:
        """Chunks academic staff directories by titles."""
        pattern = r"(?=(?:Prof\.|Doç\.|Dr\.|Öğr\.|Arş\.|Res\.))"
        chunks = re.split(pattern, text)
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
        return IndexerMain.enforce_max_length(clean_chunks)

    @staticmethod
    def split_courses(text: str) -> List[str]:
        """Chunks course plans by course codes at the start of lines."""
        pattern = r"(?=\n[A-Z]{2,4}\s*\d{3,4})"
        text = "\n" + text
        chunks = re.split(pattern, text)
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
        return IndexerMain.enforce_max_length(clean_chunks)
    
    @staticmethod
    def split_regulations_special(text: str) -> List[str]:
        """Specific chunker for disciplinary regulations capturing headers and articles."""
        pattern = r"(?=\n[A-ZİĞÜŞÖÇ\s]+cezasını gerektiren)|(?=\nMADDE\s+\d+)|(?=\n[A-ZİĞÜŞÖÇ]+\s+BÖLÜM)"
        text = "\n" + text
        chunks = re.split(pattern, text)
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
        return IndexerMain.enforce_max_length(clean_chunks)

    @staticmethod
    def split_regulations(text: str) -> List[str]:
        """Standard regulation chunking by Article (MADDE)."""
        pattern = r"(?=\nMADDE\s+\d+)"
        text = "\n" + text
        chunks = re.split(pattern, text)
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
        return IndexerMain.enforce_max_length(clean_chunks)

    @staticmethod
    def process_file(path: str, filename: str, all_chunks: List[Chunk], raw_index_map: Dict[str, List[IndexEntry]]) -> None:
        """Reads a file and applies the appropriate semantic chunking strategy."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as e:
            print(f"ERROR: Could not read file {path}. Reason: {e}")
            return

        fname = filename.lower()
        text_segments: List[str] = []
        
        # Strategy Selection
        if "disiplin" in fname:
            print(f"   -> Special Semantic Chunking (DISIPLIN) for {filename}")
            text_segments = IndexerMain.split_regulations_special(content)
        elif "akademik" in fname or "kadro" in fname:
            print(f"   -> Semantic Chunking (STAFF) for {filename}")
            text_segments = IndexerMain.split_academic_staff(content)
        elif "ders" in fname or "plan" in fname or "course" in fname:
            print(f"   -> Semantic Chunking (COURSE) for {filename}")
            text_segments = IndexerMain.split_courses(content)
        elif any(kw in fname for kw in ["yönetmelik", "yönerge", "mevzuat", "sınav"]):
            print(f"   -> Semantic Chunking (REGULATION) for {filename}")
            text_segments = IndexerMain.split_regulations(content)
        else:
            print(f"   -> Standard Paragraph Chunking for {filename}")
            paragraphs = content.split("\n\n")
            raw_segments = [p.strip() for p in paragraphs if len(p.strip()) > 10]
            text_segments = IndexerMain.enforce_max_length(raw_segments)

        doc_id = filename.replace(".txt", "")
        local_chunk_id = 0
        
        for segment in text_segments:
            # Generate embedding for vector search
            emb = get_embedding(segment)
            
            chunk = Chunk(
                docId=doc_id,
                chunkId=local_chunk_id,
                rawText=segment,
                startOffset=0,
                endOffset=len(segment),
                embedding=emb
            )
            all_chunks.append(chunk)

            # Update Keyword Index
            tokens = IndexerMain.tokenize(segment)
            seen_tokens: Set[str] = set()
            for t in tokens:
                if t in seen_tokens: continue
                seen_tokens.add(t)
                
                if t not in raw_index_map:
                    raw_index_map[t] = []
                
                raw_index_map[t].append(IndexEntry(docId=doc_id, chunkId=local_chunk_id, tf=1))
            
            local_chunk_id += 1

    @staticmethod
    def main() -> None:
        """Main entry point for the indexing process."""
        corpus_dir: str = "data/corpus"
        all_chunks: List[Chunk] = []
        raw_index_map: Dict[str, List[IndexEntry]] = {}

        print("\n=== PYTHON INDEXER STARTING (Iteration 2 - Semantic Chunks & Embeddings) ===")

        # --- STEP 1: AUTO-PURGE CACHE (NFR Requirement) --- 
        if os.path.exists(IndexerMain.CACHE_FILE):
            try:
                os.remove(IndexerMain.CACHE_FILE)
                print(f"✅ STALE CACHE PURGED: '{IndexerMain.CACHE_FILE}' removed for freshness.")
            except OSError as e:
                print(f"⚠️ WARNING: Could not purge cache file. {e}")
        
        if not os.path.exists(corpus_dir):
            print(f"CRITICAL ERROR: Corpus directory '{corpus_dir}' not found.")
            return

        files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
        for filename in files:
            print(f"Processing: {filename}")
            IndexerMain.process_file(os.path.join(corpus_dir, filename), filename, all_chunks, raw_index_map)

        print(f"=== DONE. Total Chunks: {len(all_chunks)} ===")

        try:
            os.makedirs("data", exist_ok=True)
            # Save chunks
            with open("data/chunks.json", "w", encoding="utf-8") as f:
                json.dump([c.__dict__ for c in all_chunks], f, ensure_ascii=False, indent=2)
            
            # Export keyword index
            index_export = {k: [e.__dict__ for e in v] for k, v in raw_index_map.items()}
            with open("data/index.json", "w", encoding="utf-8") as f:
                json.dump({"indexMap": index_export}, f, ensure_ascii=False, indent=2)
            
            print("Successfully saved data/chunks.json and data/index.json")
            
        except (IOError, TypeError) as e:
            print(f"CRITICAL ERROR: Could not write output files. {e}")

if __name__ == "__main__":
    IndexerMain.main()