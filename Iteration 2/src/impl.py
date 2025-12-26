import re
import copy
from typing import List, Dict, Set, Any, Optional
from src.core import IntentDetector, QueryWriter, Retriever, Reranker, AnswerAgent
from src.models import Intent, Hit, KeywordIndex, Answer, Citation, Chunk
from src.utils import get_embedding, cosine_similarity

# --- 1. INTENT DETECTOR ---
class ConfigurableIntentDetector(IntentDetector):
    def __init__(self, rules: Dict[str, List[str]]):
        self.rules = rules

    def detect(self, question: str) -> Intent:
        try:
            if not question: return Intent.UNKNOWN
            q = question.lower()
            for intent_name, keywords in self.rules.items():
                if any(k in q for k in keywords):
                    try:
                        return Intent[intent_name]
                    except (KeyError, ValueError):
                        continue
        except Exception:
            pass
        return Intent.UNKNOWN

# --- 2. QUERY WRITER ---
class HeuristicQueryWriter(QueryWriter):
    STOP_WORDS = {
        "nedir", "kimdir", "nasıl", "nerede", "hangi", "kaç", "mi", "mı", "mu", "mü", "soru",
        "ve", "ile", "için", "bu", "şu", "o", "bir", "var", "yok", "veya", "olarak",
        "ders", "dersi", "dersinin", "hakkında", "bilgi", "ilgili", "kısmı", "bölüm", "mühendisliği",
        "in", "ın", "un", "ün", "nin", "nın", "nun", "nün", "yi", "yı", "yu", "yü"
    }
    
    SYNONYMS = {
        "çap": ["çift anadal", "ikinci anadal", "madde 35"],
        "yandal": ["yan dal"],
        "staj": ["pratik çalışma", "zorunlu staj", "iş günü"],
        "kalmak": ["başarısız", "tekrar", "ff", "dersten kalma", "alt limit"],
        "dondurma": ["kayıt dondurma", "izinli sayılma", "haklı ve geçerli neden"],
        "yurt dışı": ["erasmus", "farabi", "değişim", "anlaşmalı üniversite"],
        "af": ["öğrenci affı"],
        "yaz okulu": ["yaz öğretimi", "başka üniversiteden ders"],
        "büt": ["bütünleme"],
        "tek ders": ["mezuniyet sınavı", "tek ders sınavı"],
        "diploma kayıp": ["duplikata", "yeniden düzenleme"],
        "ofis": ["oda", "yer", "nerede", "iletişim", "e-posta"],
        "ön koşul": ["prerequisite", "önkoşul", "condition"]
    }

    def write(self, question: str, intent: Intent) -> List[str]:
        terms = []
        try:
            q_lower = question.lower()
            normalized = re.sub(r'[^\w\s]', ' ', q_lower)
            tokens = normalized.split()
            
            for token in tokens:
                if len(token) < 2: continue
                if token not in self.STOP_WORDS or any(char.isdigit() for char in token):
                    terms.append(token)
            
            for key, syn_list in self.SYNONYMS.items():
                if key in q_lower:
                    terms.extend(syn_list)
        except Exception:
            pass
        return terms

# --- 3. RETRIEVER ---
class KeywordRetriever(Retriever):
    def retrieve(self, query_terms: List[str], index: KeywordIndex) -> List[Hit]:
        hits = []
        try:
            score_map: Dict[str, float] = {}
            match_counts: Dict[str, Set[str]] = {}
            
            for term in query_terms:
                if term in index.indexMap:
                    entries = index.indexMap[term]
                    for entry in entries:
                        key = f"{entry.docId}::{entry.chunkId}"
                        score_map[key] = score_map.get(key, 0.0) + entry.tf
                        if key not in match_counts: match_counts[key] = set()
                        match_counts[key].add(term)
            
            for key, tf_score in score_map.items():
                try:
                    doc_id, chunk_id_str = key.split("::")
                    chunk_id = int(chunk_id_str)
                    distinct_matches = len(match_counts[key])
                    final_score = (distinct_matches * 1000.0) + tf_score
                    hits.append(Hit(doc_id, chunk_id, final_score, None))
                except (ValueError, KeyError, IndexError): 
                    continue
                    
            hits.sort()
        except Exception:
            pass
        return hits

# --- 4. RERANKERS ---

class SimpleReranker(Reranker):
    def __init__(self, all_chunks: List[Chunk]):
        self.chunk_map = {f"{c.docId}_{c.chunkId}": c for c in all_chunks}

    def rerank(self, query_terms: List[str], hits: List[Hit]) -> List[Hit]:
        try:
            for hit in hits:
                try:
                    key = f"{hit.docId}_{hit.chunkId}"
                    chunk = self.chunk_map.get(key)
                    if not chunk: continue

                    hit.chunkText = chunk.rawText
                    text = hit.chunkText.lower()

                    tf_sum = sum(text.count(t.lower()) for t in query_terms)

                    positions = []
                    for term in query_terms:
                        idx = text.find(term.lower())
                        if idx != -1: positions.append(idx)

                    proximity_bonus = 0
                    if len(positions) >= 2:
                        positions.sort()
                        for i in range(len(positions) - 1):
                            if abs(positions[i] - positions[i+1]) <= 15:
                                proximity_bonus = 5
                                break

                    title_boost = 0
                    if any(t.lower() in hit.docId.lower() for t in query_terms):
                        title_boost = 3

                    silver_bullet = 0
                    for t in query_terms:
                        if any(char.isdigit() for char in t):  
                            if t.lower() in text:              
                                silver_bullet = 200            
                                break

                    hit.score = (tf_sum * 10) + proximity_bonus + title_boost + silver_bullet
                    hit.sort_index = (-hit.score, hit.docId, hit.chunkId)
                except Exception:
                    continue
            hits.sort()
        except Exception:
            pass
        return hits

class CosineReranker(Reranker):
    def __init__(self, all_chunks: List[Chunk]):
        self.chunk_map = {f"{c.docId}_{c.chunkId}": c for c in all_chunks}

    def rerank(self, query_tokens: List[str], hits: List[Hit]) -> List[Hit]:
        try:
            query_str = " ".join(query_tokens)
            query_vec = get_embedding(query_str)
            
            critical_terms = [t.lower() for t in query_tokens if len(t) > 3 or any(c.isdigit() for c in t)]

            target_course_code = None
            code_match = re.search(r"\b([A-Z]{3,4}\s?\d{3,4})\b", query_str.upper())
            if code_match:
                target_course_code = code_match.group(1).replace(" ", "")

            is_tek_ders = "tek ders" in query_str.lower()
            is_cap = "çap" in query_str.lower() or "çift anadal" in query_str.lower()
            is_yatay_gecis = "yatay geçiş" in query_str.lower()

            for hit in hits:
                try:
                    vec = hit.embedding
                    c = self.chunk_map.get(f"{hit.docId}_{hit.chunkId}")
                    
                    if not vec and c:
                        vec = c.embedding if c.embedding else get_embedding(c.rawText)
                        hit.chunkText = c.rawText
                        hit.embedding = vec 
                    elif c and not hit.chunkText:
                        hit.chunkText = c.rawText
                    
                    base_score = 0.0
                    if vec and query_vec:
                        base_score = cosine_similarity(query_vec, vec) * 100.0

                    boost = 0
                    if hit.chunkText:
                        text_upper = hit.chunkText.strip().upper()
                        doc_id = hit.docId.lower()
                        
                        if target_course_code and text_upper.startswith(target_course_code):
                            boost += 300.0 
                        
                        if any(t in doc_id for t in critical_terms):
                            boost += 50.0 

                        matches = sum(1 for term in critical_terms if term in hit.chunkText.lower())
                        boost += (matches * 10.0)

                        if is_tek_ders:
                            if "tek" in doc_id and "ders" in doc_id: boost += 500.0
                            elif "çap" in doc_id or "yatay" in doc_id: boost -= 200.0
                        elif is_cap:
                            if "çap" in doc_id or "anadal" in doc_id: boost += 300.0
                        elif is_yatay_gecis:
                            if "yatay" in doc_id: boost += 300.0

                    hit.score = base_score + boost
                    hit.sort_index = (-hit.score, hit.docId, hit.chunkId)
                except Exception:
                    continue
            hits.sort(key=lambda h: h.score, reverse=True)
        except Exception:
            pass
        return hits

# --- 5. ANSWER AGENTS ---

class KeywordAnswerAgent(AnswerAgent):
    def answer(self, question: str, top_hits: List[Hit]) -> Answer:
        try:
            if not top_hits: return Answer("Bilgi bulunamadı.", [])
            best_hit = top_hits[0]
            text = best_hit.chunkText or ""
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 2]
            doc_id = best_hit.docId.lower()
            
            if "ders" in doc_id or "plan" in doc_id:
                course_code_match = re.search(r"([A-Z]{3,4}\s?\d{3,4})", question.upper())
                if course_code_match:
                    target_code = course_code_match.group(1).replace(" ", "")
                    for line in lines:
                        if line.replace(" ", "").upper().startswith(target_code):
                            return Answer(line, [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])

            if "akademik" in doc_id or "kadro" in doc_id:
                titles = ["prof", "doç", "dr.", "öğr", "arş", "gör"]
                name_line = next((l for l in lines if any(l.lower().startswith(t) for t in titles)), None)
                if name_line:
                    info_lines = [name_line]
                    for line in lines:
                        if line == name_line: continue
                        if any(x in line.lower() for x in ["ofis", "office", "m2", "bina", "e-posta", "@", "tel:", "bs:", "ms:", "phd:"]):
                            info_lines.append(line)
                    return Answer("\n".join(info_lines), [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])

            return Answer(text, [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])
        except Exception:
            return Answer("Cevap oluşturulurken bir hata oluştu.", [])

class VectorAnswerAgent(AnswerAgent):
    def answer(self, question: str, top_hits: List[Hit]) -> Answer:
        try:
            if not top_hits: return Answer("Bilgi bulunamadı.", [])
            best_hit = top_hits[0]
            chunk_text = best_hit.chunkText or ""
            lines = [l.strip() for l in chunk_text.split("\n") if len(l.strip()) > 2]
            
            if not lines: return Answer(chunk_text, [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])

            doc_id = best_hit.docId.lower()
            q_vec = get_embedding(question)

            best_idx = 0
            max_score = -1.0
            found_strict_match = False
            
            course_code_match = re.search(r"([A-Z]{3,4}\s?\d{3,4})", question.upper())
            if course_code_match and ("ders" in doc_id or "plan" in doc_id):
                target_code = course_code_match.group(1).replace(" ", "")
                for i, line in enumerate(lines):
                    if line.replace(" ", "").upper().startswith(target_code):
                        best_idx = i
                        found_strict_match = True
                        break 

            if not found_strict_match and ("akademik" in doc_id or "kadro" in doc_id):
                titles = ["prof", "doç", "dr.", "öğr", "arş", "gör"]
                for i, line in enumerate(lines):
                    if any(line.lower().startswith(t) for t in titles):
                        q_slugs = question.lower().split()
                        if sum(1 for s in q_slugs if s in line.lower() and len(s)>3) >= 1:
                            best_idx = i
                            found_strict_match = True 
                            break
            
            if not found_strict_match:
                for i, line in enumerate(lines):
                    try:
                        s = cosine_similarity(q_vec, get_embedding(line))
                        if s > max_score:
                            max_score = s
                            best_idx = i
                    except Exception: continue

            start_idx = max(0, best_idx - 2)
            end_idx = min(len(lines), best_idx + 8)
            context_lines = lines[start_idx:end_idx]

            if "akademik" in doc_id or "kadro" in doc_id:
                q_lower = question.lower()
                filtered_academic = []
                titles = ["prof", "doç", "dr.", "öğr", "arş", "gör"]
                name_line = next((l for l in context_lines if any(l.lower().startswith(t) for t in titles)), None)
                if name_line: filtered_academic.append(name_line)
                
                found_spec = False
                if any(k in q_lower for k in ["nerede", "ofis", "oda"]):
                    for l in context_lines:
                        if any(x in l.lower() for x in ["ofis", "office", "m2", "bina"]):
                            filtered_academic.append(l); found_spec = True
                elif any(k in q_lower for k in ["mail", "e-posta", "iletişim"]):
                    for l in context_lines:
                        if "@" in l: filtered_academic.append(l); found_spec = True
                
                if found_spec:
                    return Answer("\n".join(filtered_academic), [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])

            if context_lines and not context_lines[-1].strip().endswith((".", ":", ";", "?", "!")):
                if end_idx < len(lines): context_lines.append(lines[end_idx])

            filtered_lines = []
            q_words = set(question.lower().split())
            for line in context_lines:
                if any(x in line for x in ["MADDE", "Yönerge", "Önkoşul"]):
                    filtered_lines.append(line)
                else:
                    intersection = set(line.lower().split()).intersection(q_words)
                    if len(intersection) > 0 or len(line.split()) < 4:
                        filtered_lines.append(line)

            final_text = "\n".join(filtered_lines) if filtered_lines else "\n".join(context_lines)
            return Answer(final_text, [Citation(best_hit.docId, f"Chunk{best_hit.chunkId}", 0, 0)])
        except Exception:
            return Answer("Cevap oluşturulurken teknik bir hata oluştu.", [])