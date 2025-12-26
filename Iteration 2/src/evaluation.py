import time
import re
from typing import List, Dict, Any
from src.models import Answer

class RagEvaluator:
    """
    RAG boru hattının sistematik değerlendirmesini yapar.
    Hem doküman hem de parça (chunk) düzeyinde doğruluğu ölçer.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_item(self, question: str, expected_doc: str, expected_chunk_id: int = None) -> Dict[str, Any]:
        """
        Tek bir soruyu beklenen sonuçlarla karşılaştırır.
        """
        start_time = time.time()
        answer: Answer = self.pipeline.run(question)
        latency_ms = (time.time() - start_time) * 1000

        retrieved_citations = answer.citations or []
        
        doc_match = False
        chunk_match = False
        top_doc = "NONE"
        top_chunk = "NONE"

        if retrieved_citations:
            top_citation = retrieved_citations[0]
            top_doc = top_citation.docId.lower()
            top_chunk = top_citation.sectionId # Örn: "Chunk4" veya "akademik kadro_Chunk4"

            # 1. Doküman Seviyesinde Doğruluk (Top-1)
            if expected_doc.lower() in top_doc:
                doc_match = True

            # 2. Parça (Chunk) Seviyesinde Doğruluk (Top-1)
            if expected_chunk_id is not None:
                # Regex ile "Chunk4" veya "..._Chunk4" içindeki 4 rakamını bulur
                # 'Chunk' kelimesinden sonra gelen rakamları yakalar
                match = re.search(r'Chunk(\d+)', top_chunk, re.IGNORECASE)
                if match:
                    found_id = int(match.group(1))
                    # Hem doküman hem de parça numarası doğru olmalı
                    if found_id == expected_chunk_id and doc_match:
                        chunk_match = True

        # 3. Coverage@5 (Doğru doküman ilk 5 kaynakta var mı?)
        retrieved_docs = [c.docId.lower() for c in retrieved_citations[:5]]
        coverage_at_5 = any(expected_doc.lower() in d for d in retrieved_docs)

        return {
            "latency": latency_ms,
            "doc_match": doc_match,
            "chunk_match": chunk_match,
            "coverage_at_5": coverage_at_5,
            "top_doc": top_doc,
            "top_chunk": top_chunk
        }