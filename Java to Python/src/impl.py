import re
from typing import List, Dict, Set
from .core import IntentDetector, QueryWriter, Retriever, Reranker, AnswerAgent
from .models import Intent, Hit, KeywordIndex, IndexEntry, Answer, Citation, Chunk

class RuleBasedIntentDetector(IntentDetector):
    def detect(self, question: str) -> Intent:
        q = question.lower()
        
        if any(k in q for k in ["hoca", "ofis", "mail", "iletişim", "kimdir", "başkan"]):
            return Intent.STAFF_LOOKUP
        
        if any(k in q for k in ["ders", "kredi", "ects", "akts", "önkoşul", "dönem"]):
            return Intent.COURSE_INFO
            
        if any(k in q for k in ["yönetmelik", "yönerge", "sınav", "staj", "mezuniyet", "çap", "yatay"]):
            return Intent.POLICY_FAQ
            
        if any(k in q for k in ["kayıt", "dondurma", "harç"]):
            return Intent.REGISTRATION
            
        return Intent.UNKNOWN

class HeuristicQueryWriter(QueryWriter):
    STOP_WORDS = {
        "nedir", "kimdir", "nasıl", "nerede", "hangi", "kaç", "mi", "mı", "mu", "mü", "soru",
        "ve", "ile", "için", "bu", "şu", "o", "bir", "var", "yok", "veya", "olarak",
        "ders", "dersi", "dersinin", "hakkında", "bilgi", "ilgili", "kısmı", "bölüm", "mühendisliği"
    }

    def write(self, question: str, intent: Intent) -> List[str]:
        terms = []
        normalized = question.lower()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        tokens = normalized.split()
        
        for token in tokens:
            if len(token) < 2: continue
            if token not in self.STOP_WORDS or any(char.isdigit() for char in token):
                terms.append(token)
        
        if intent == Intent.STAFF_LOOKUP and "ofis" not in terms:
            terms.append("ofis")
                
        return terms

class KeywordRetriever(Retriever):
    def retrieve(self, query_terms: List[str], index: KeywordIndex) -> List[Hit]:
        score_map: Dict[str, float] = {}
        match_counts: Dict[str, Set[str]] = {}
        
        for term in query_terms:
            if term in index.indexMap:
                entries = index.indexMap[term]
                for entry in entries:
                    # build unique key for docId::chunkId
                    key = f"{entry.docId}::{entry.chunkId}"
                    
                    score_map[key] = score_map.get(key, 0.0) + entry.tf
                    
                    if key not in match_counts:
                        match_counts[key] = set()
                    match_counts[key].add(term)
        
        hits = []
        for key, tf_score in score_map.items():
            doc_id, chunk_id_str = key.split("::")
            
            try:
                chunk_id = int(chunk_id_str)
            except (ValueError, TypeError):
                # Skip if None, empty string, "null", or otherwise invalid
                continue

            distinct_matches = len(match_counts[key])
            final_score = (distinct_matches * 1000.0) + tf_score
            
            hits.append(Hit(doc_id, chunk_id, final_score, None))
            
        hits.sort()
        return hits


class SimpleReranker(Reranker):
    def __init__(self, all_chunks: List[Chunk]):
        self.chunk_map = {f"{c.docId}_{c.chunkId}": c for c in all_chunks}

    def rerank(self, query_terms: List[str], hits: List[Hit]) -> List[Hit]:
        for hit in hits:
            key = f"{hit.docId}_{hit.chunkId}"
            chunk = self.chunk_map.get(key)
            if not chunk:
                continue

            hit.chunkText = chunk.rawText
            text = hit.chunkText.lower()

            # --- TF SUM ---
            tf_sum = sum(text.count(t.lower()) for t in query_terms)

            # --- PROXIMITY BONUS ---
            positions = []
            for term in query_terms:
                idx = text.find(term.lower())
                if idx != -1:
                    positions.append(idx)

            proximity_bonus = 0
            if len(positions) >= 2:
                positions.sort()
                for i in range(len(positions) - 1):
                    if abs(positions[i] - positions[i+1]) <= 15:
                        proximity_bonus = 5
                        break

            # --- TITLE BOOST ---
            title_boost = 0
            if any(t.lower() in hit.docId.lower() for t in query_terms):
                title_boost = 3

            # --- SILVER BULLET BOOST (like CSE3063 ) ---
            silver_bullet = 0
            for t in query_terms:
                if any(char.isdigit() for char in t):  
                    if t.lower() in text:              
                        silver_bullet = 200            
                        break

            # --- FINAL SCORE ---
            hit.score = (tf_sum * 10) + proximity_bonus + title_boost + silver_bullet

            hit.sort_index = (-hit.score, hit.docId, hit.chunkId)

        hits.sort()
        return hits


import re
from typing import List
from .core import AnswerAgent
from .models import Answer, Citation, Hit
import locale

class TemplateAnswerAgent(AnswerAgent):

    def answer(self, question: str, top_hits: List[Hit]) -> Answer:
        # 1. Fallback
        if not top_hits:
            return Answer(
                "Sorry, I could not find any information regarding this.",
                []
            )

        best_hit = top_hits[0]
        raw_text = (best_hit.chunkText or "").replace("\\n", "\n")

        lines = raw_text.split("\n")

        # Query normalization
        q_norm = re.sub(r"[^0-9A-Za-zğüşöçıİĞÜŞÖÇ ]", " ",
                        question.lower())
        query_terms = q_norm.split()

        # Check digit presence (course code)
        has_digit = any(re.search(r"\d", t) for t in query_terms)

        target_indices = []

        # 3. Line selection strategy
        if has_digit:
            # search lines containing digit terms
            for i, ln in enumerate(lines):
                lower = ln.lower()
                for t in query_terms:
                    if re.search(r"\d", t) and t in lower:
                        target_indices.append(i)
        else:
            # keyword-density strategy
            best_idx = -1
            max_score = -1

            for i, ln in enumerate(lines):
                lower = ln.lower()
                score = 0
                for t in query_terms:
                    if len(t) > 2 and t in lower:
                        score += 1
                if score > max_score:
                    max_score = score
                    best_idx = i

            if best_idx != -1:
                target_indices.append(best_idx)

        # 4. Context Formatting
        final_out = []
        processed = set()

        for idx in target_indices:
            if idx < 0 or idx >= len(lines):
                continue

            line = lines[idx].strip()
            if not line:
                continue

            is_course_line = bool(re.match(r"^\s*[A-Z]{2,}\s*\d{3,}", line))

            if is_course_line:
                # Look back for "Semester/Yarıyıl"
                for k in range(idx - 1, max(-1, idx - 6) - 1, -1):
                    if k < 0:
                        break

                    prev = lines[k].lower()
                    if ("dönem" in prev or
                        "semester" in prev or
                        "yarıyıl" in prev):
                        if k not in processed:
                            final_out.append(lines[k].strip())
                            processed.add(k)
                        break

                # Add course line itself
                if idx not in processed:
                    final_out.append(line)
                    processed.add(idx)

            else:
                # Expand context block
                start = idx
                end = idx

                # upward expansion
                while start > 0:
                    p = lines[start - 1].strip()
                    if p == "" or self._is_title_line(p):
                        break
                    start -= 1

                # downward expansion
                while end < len(lines) - 1:
                    n = lines[end + 1].strip()
                    if n == "" or self._is_title_line(n):
                        break
                    end += 1

                for k in range(start, end + 1):
                    if k not in processed:
                        final_out.append(lines[k].strip())
                        processed.add(k)

        # 5. Finalize text
        final_text = "\n".join(final_out).strip()
        if not final_text:
            # Fallback to first 300 chars of raw text
            final_text = raw_text[:300] + ("..." if len(raw_text) > 300 else "")

        # 6. Citation
        citations = [
            Citation(
                docId=best_hit.docId,
                sectionId=f"P{best_hit.chunkId}",
                startOffset=0,
                endOffset=0
            )
        ]

        return Answer(final_text, citations)

    # ===== Helper =====
    def _is_title_line(self, line: str) -> bool:
        l = line.lower()
        return (
            l.startswith("prof") or
            l.startswith("doç") or
            l.startswith("dr") or
            l.startswith("öğr") or
            l.startswith("arş")
        )
