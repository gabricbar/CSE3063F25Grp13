import unittest
from unittest.mock import MagicMock, patch, Mock
from src.models import Intent, KeywordIndex, IndexEntry, Chunk, Hit, Answer, Citation
from src.impl import (
    ConfigurableIntentDetector,
    HeuristicQueryWriter,
    KeywordRetriever,
    SimpleReranker,
    CosineReranker,
    VectorAnswerAgent,
    KeywordAnswerAgent
)

# ============================================================================
# 1. TEST BASE CLASS - OOPS Prensipleri: Inheritance & Encapsulation
# ============================================================================
class BaseRagTestCase(unittest.TestCase):
    """Tüm testlerin kalıtacağı temel sınıf - Reusability & DRY"""
    
    def setUp(self):
        """Test ortamını hazırlar. Her testten önce çalışır."""
        self._initialize_chunks()
        self._initialize_rules()
        self._initialize_index()

    def _initialize_chunks(self):
        """Chunk nesnelerini ayarla"""
        self.chunk1 = Chunk(
            docId="ders_planı.txt", 
            chunkId=0, 
            rawText="CSE3063 Object Oriented Design. Önkoşul: CSE1242", 
            startOffset=0, 
            endOffset=50, 
            embedding=[0.1]*384
        )
        self.chunk2 = Chunk(
            docId="yonetmelik.txt", 
            chunkId=1, 
            rawText="MADDE 1: Sınavlar yüz yüze yapılır.", 
            startOffset=0, 
            endOffset=35, 
            embedding=[0.2]*384
        )
        self.chunk_staff = Chunk(
            docId="akademik_kadro.txt", 
            chunkId=2, 
            rawText="Prof. Dr. Mustafa AĞAOĞLU\nOfis: M2-221\nTelefon: 3540", 
            startOffset=0, 
            endOffset=50, 
            embedding=[0.3]*384
        )
        self.chunks = [self.chunk1, self.chunk2, self.chunk_staff]

    def _initialize_rules(self):
        """Intent kurallarını ayarla"""
        self.rules = {
            "STAFF_LOOKUP": ["hoca", "ofis", "nerede"],
            "COURSE_INFO": ["ders", "kredi", "önkoşul"],
            "POLICY_FAQ": ["yönetmelik"],
            "REGISTRATION": ["kayıt", "dönem"],
            "UNKNOWN": []
        }

    def _initialize_index(self):
        """Keyword indexini ayarla"""
        self.index = KeywordIndex()
        self.index.indexMap["cse3063"] = [IndexEntry("ders_planı.txt", 0, 2)]
        self.index.indexMap["object"] = [IndexEntry("ders_planı.txt", 0, 1)]
        self.index.indexMap["oriented"] = [IndexEntry("ders_planı.txt", 0, 1)]
        self.index.indexMap["design"] = [IndexEntry("ders_planı.txt", 0, 1)]
        self.index.indexMap["madde"] = [IndexEntry("yonetmelik.txt", 1, 3)]
        self.index.indexMap["sınav"] = [IndexEntry("yonetmelik.txt", 1, 1)]


class RagSystemTest(BaseRagTestCase):
    """OOPS: Inheritance - BaseRagTestCase'den kalıtım"""


    # ========================================================================
    # TEST 1: Intent Detection - Polymorphism Uygulanmış
    # ========================================================================
    def test_intent_detection_course_info(self):
        """Intent COURSE_INFO'nun doğru tanındığını test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("Bu dersin kredisi kaç?")
        self.assertEqual(result, Intent.COURSE_INFO)

    def test_intent_detection_staff_lookup(self):
        """Intent STAFF_LOOKUP'ün doğru tanındığını test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("Murat hoca nerede?")
        self.assertEqual(result, Intent.STAFF_LOOKUP)

    def test_intent_detection_policy_faq(self):
        """Intent POLICY_FAQ'ün doğru tanındığını test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("Yönetmelik hakkında bilgi")
        self.assertEqual(result, Intent.POLICY_FAQ)

    def test_intent_detection_unknown(self):
        """Intent UNKNOWN'ın döndürüldüğünü test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("Hava durumu nasıl?")
        self.assertEqual(result, Intent.UNKNOWN)

    def test_intent_detection_registration(self):
        """Intent REGISTRATION'ın doğru tanındığını test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("Kayıt döneminde neler yapabilirim?")
        self.assertEqual(result, Intent.REGISTRATION)

    # ========================================================================
    # TEST 2: Query Writer - Encapsulation (Private STOP_WORDS)
    # ========================================================================
    def test_query_writer_stopword_removal(self):
        """Stopword'lerin çıkarıldığını test eder"""
        writer = HeuristicQueryWriter()
        terms = writer.write("CSE3063 ve CSE1242 nedir?", Intent.UNKNOWN)
        
        self.assertIn("cse3063", terms)
        self.assertIn("cse1242", terms)
        self.assertNotIn("ve", terms)
        self.assertNotIn("nedir", terms)

    def test_query_writer_synonym_expansion(self):
        """Synonym genişletmesinin çalıştığını test eder"""
        writer = HeuristicQueryWriter()
        terms = writer.write("çap yapmak istiyorum", Intent.POLICY_FAQ)
        
        self.assertTrue(any("çift" in t or "anadal" in t for t in terms))

    def test_query_writer_booster_staff_lookup(self):
        """Staff Lookup için booster kelimelerinin eklendiğini test eder"""
        writer = HeuristicQueryWriter()
        terms = writer.write("Mustafa Hoca", Intent.STAFF_LOOKUP)
        
        self.assertIn("mustafa", terms)
        self.assertIn("hoca", terms)

    def test_query_writer_numeric_tokens(self):
        """Sayısal tokenlerin korunduğunu test eder"""
        writer = HeuristicQueryWriter()
        terms = writer.write("CSE3063 kodu nedir", Intent.COURSE_INFO)
        
        self.assertTrue(any("3063" in t for t in terms))

    def test_query_writer_short_token_exclusion(self):
        """Çok kısa tokenlerin (< 2 kar) hariç tutulduğunu test eder"""
        writer = HeuristicQueryWriter()
        terms = writer.write("a b test", Intent.UNKNOWN)
        
        # "a" ve "b" atılmalı
        self.assertNotIn("a", terms)
        self.assertNotIn("b", terms)

    # ========================================================================
    # TEST 3: Keyword Retriever - Polymorphism (Retriever Interface)
    # ========================================================================
    def test_retriever_distinct_matches_priority(self):
        """Farklı kelime eşleşmesinin önceliklendirildiğini test eder"""
        retriever = KeywordRetriever()
        
        # Doc1: "java" ve "oop" (2 distinct match)
        # Doc2: "java" 10 kere (1 distinct match)
        index = KeywordIndex()
        index.indexMap["java"] = [
            IndexEntry("doc1", 0, 1),
            IndexEntry("doc2", 0, 10)
        ]
        index.indexMap["oop"] = [IndexEntry("doc1", 0, 1)]
        
        hits = retriever.retrieve(["java", "oop"], index)
        
        # doc1 doc2'den önce gelmelidir
        self.assertEqual(hits[0].docId, "doc1")
        self.assertTrue(hits[0].score > hits[1].score)

    def test_retriever_empty_index(self):
        """Boş indexte boş sonuç döndürülmesini test eder"""
        retriever = KeywordRetriever()
        index = KeywordIndex()
        
        hits = retriever.retrieve(["nonexistent"], index)
        self.assertEqual(len(hits), 0)

    def test_retriever_single_term(self):
        """Tek terim aramasını test eder"""
        retriever = KeywordRetriever()
        
        hits = retriever.retrieve(["cse3063"], self.index)
        
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].docId, "ders_planı.txt")

    def test_retriever_multi_term_query(self):
        """Çok terimli sorguyu test eder"""
        retriever = KeywordRetriever()
        
        hits = retriever.retrieve(["cse3063", "object", "design"], self.index)
        
        self.assertGreater(len(hits), 0)
        # 3 matching terms olduğu için yüksek score olmalı
        self.assertGreater(hits[0].score, 1000)

    # ========================================================================
    # TEST 4: Simple Reranker - Encapsulation & Abstraction
    # ========================================================================
    def test_reranker_title_boost(self):
        """Başlık bonusunun verildiğini test eder"""
        reranker = SimpleReranker(self.chunks)
        
        hit1 = Hit("ders_planı.txt", 0, 1.0, None)
        hit2 = Hit("yonetmelik.txt", 1, 1.0, None)
        
        hits = reranker.rerank(["ders"], [hit1, hit2])
        
        # ders_planı.txt başlığında "ders" var, bonus almalı
        self.assertEqual(hits[0].docId, "ders_planı.txt")

    def test_reranker_tf_calculation(self):
        """TF hesaplamasının doğru yapıldığını test eder"""
        reranker = SimpleReranker(self.chunks)
        
        hit = Hit("ders_planı.txt", 0, 1.0, None)
        reranked = reranker.rerank(["cse3063"], [hit])
        
        # Chunk metninde "CSE3063" 1 kere geçiyor, TF = 1
        # Score = (TF * 10) = 10
        self.assertGreater(reranked[0].score, 0)

    def test_reranker_silver_bullet_boost(self):
        """Sayısal kod boostu test eder"""
        reranker = SimpleReranker(self.chunks)
        
        hit = Hit("ders_planı.txt", 0, 1.0, None)
        reranked = reranker.rerank(["cse3063"], [hit])
        
        # CSE3063 sayısal token, 200 bonus almalı
        self.assertGreater(reranked[0].score, 200)

    def test_reranker_proximity_bonus(self):
        """Kelimeler yakın olduğunda bonus test eder"""
        chunk = Chunk(
            docId="test.txt", 
            chunkId=0, 
            rawText="java ve oop programlama",  # 8 karakterlik aralık
            startOffset=0, 
            endOffset=25,
            embedding=[0.1]*384
        )
        reranker = SimpleReranker([chunk])
        
        hit = Hit("test.txt", 0, 1.0, None)
        reranked = reranker.rerank(["java", "oop"], [hit])
        
        # "java" ve "oop" 8 char aralıkta, proximity bonus almalı
        self.assertGreater(reranked[0].score, 20)

    # ========================================================================
    # TEST 5: Cosine Reranker - Polymorphism
    # ========================================================================
    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_cosine_reranker_definition_priority(self, mock_sim, mock_emb):
        """Kurs tanımının referanstan önce gelmesini test eder"""
        mock_emb.return_value = [0.1] * 384
        mock_sim.return_value = 0.5
        
        chunk1 = Chunk(
            docId="ders_planı.txt",
            chunkId=0,
            rawText="CSE3063 Object Oriented Design\nÖnkoşul: CSE1242",
            startOffset=0,
            endOffset=40,
            embedding=[0.1]*384
        )
        chunk2 = Chunk(
            docId="yönetmelik.txt",
            chunkId=1,
            rawText="CSE3063 sınav kuralları",
            startOffset=0,
            endOffset=20,
            embedding=[0.2]*384
        )
        
        reranker = CosineReranker([chunk1, chunk2])
        
        hit1 = Hit("ders_planı.txt", 0, 50.0, "CSE3063 Object Oriented Design\nÖnkoşul: CSE1242")
        hit1.embedding = [0.1]*384
        hit2 = Hit("yönetmelik.txt", 1, 50.0, "CSE3063 sınav kuralları")
        hit2.embedding = [0.2]*384
        
        reranked = reranker.rerank(["cse3063"], [hit1, hit2])
        
        # hit1 (definition) hit2'den önce gelmelidir
        self.assertEqual(reranked[0].docId, "ders_planı.txt")

    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_cosine_reranker_tek_ders_protection(self, mock_sim, mock_emb):
        """Tek Ders sorgusu için koruma test eder"""
        mock_emb.return_value = [0.1] * 384
        mock_sim.return_value = 0.5
        
        chunk_tek = Chunk(
            docId="tek_ders_sınav.txt",
            chunkId=0,
            rawText="Tek Ders Sınavı Yönetmeliği",
            startOffset=0,
            endOffset=30,
            embedding=[0.1]*384
        )
        chunk_cap = Chunk(
            docId="cap_yonetmeligii.txt",
            chunkId=1,
            rawText="Çift Anadal Programı",
            startOffset=0,
            endOffset=20,
            embedding=[0.2]*384
        )
        
        reranker = CosineReranker([chunk_tek, chunk_cap])
        
        hit_tek = Hit("tek_ders_sınav.txt", 0, 50.0, "Tek Ders Sınavı Yönetmeliği")
        hit_tek.embedding = [0.1]*384
        hit_cap = Hit("cap_yonetmeligii.txt", 1, 50.0, "Çift Anadal Programı")
        hit_cap.embedding = [0.2]*384
        
        reranked = reranker.rerank(["tek", "ders"], [hit_cap, hit_tek])
        
        # tek ders sorgusu için tek_ders_sınav.txt önce gelmelidir
        self.assertEqual(reranked[0].docId, "tek_ders_sınav.txt")

    # ========================================================================
    # TEST 6: Keyword Answer Agent - Encapsulation
    # ========================================================================
    def test_keyword_answer_agent_basic(self):
        """KeywordAnswerAgent temel cevap döndürüşünü test eder"""
        agent = KeywordAnswerAgent()
        hit = Hit("ders_planı.txt", 0, 100.0, "CSE3063 Object Oriented Design")
        
        answer = agent.answer("CSE3063 nedir", [hit])
        
        self.assertIsNotNone(answer.finalText)
        self.assertGreater(len(answer.citations), 0)
        self.assertEqual(answer.citations[0].docId, "ders_planı.txt")

    def test_keyword_answer_agent_empty_hits(self):
        """Boş hits listesi test eder"""
        agent = KeywordAnswerAgent()
        
        answer = agent.answer("soru", [])
        
        self.assertEqual(answer.finalText, "Bilgi bulunamadı.")

    def test_keyword_answer_agent_course_code_protection(self):
        """Kurs koduna özel koruma test eder"""
        agent = KeywordAnswerAgent()
        chunk_text = "CSE3063 Object Oriented Design\nÖnkoşul: CSE1242"
        hit = Hit("ders_planı.txt", 0, 100.0, chunk_text)
        
        answer = agent.answer("CSE3063 nedir", [hit])
        
        # CSE3063 ile başlayan satır seçilmeli
        self.assertIn("Object Oriented Design", answer.finalText)

    def test_keyword_answer_agent_staff_extraction(self):
        """Staff bilgisini extraction test eder"""
        agent = KeywordAnswerAgent()
        chunk_text = "Prof. Dr. Mustafa AĞAOĞLU\nOfis: M2-221\nTelefon: 3540"
        hit = Hit("akademik_kadro.txt", 0, 100.0, chunk_text)
        
        answer = agent.answer("Prof hakkında bilgi", [hit])
        
        # Staff bulunmuşsa atıf eklenmeli
        self.assertGreater(len(answer.citations), 0)

    # ========================================================================
    # TEST 7: Vector Answer Agent - Polymorphism & Abstraction
    # ========================================================================
    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_vector_agent_course_code_strict_match(self, mock_sim, mock_emb):
        """Kurs kodunda strict match test eder"""
        mock_emb.return_value = [0.1] * 384
        mock_sim.return_value = 0.5
        
        agent = VectorAnswerAgent()
        chunk_text = "CSE3063 Object Oriented Design\nÖnkoşul: CSE1242"
        hit = Hit("ders_planı.txt", 0, 100.0, chunk_text)
        hit.embedding = [0.1]*384
        
        answer = agent.answer("CSE3063 nedir", [hit])
        
        self.assertIn("Object Oriented Design", answer.finalText)

    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_vector_agent_context_expansion(self, mock_sim, mock_emb):
        """Context window genişletmesini test eder"""
        mock_emb.return_value = [0.1] * 384
        # En yüksek benzerlik 2. satırda (d şıkkı)
        # Başlık dahil olması için en az 2 satır geriye gitmeli (start_idx >= 0)
        mock_sim.side_effect = [0.1, 0.1, 0.9, 0.1, 0.1]
        
        agent = VectorAnswerAgent()
        text = "Uzaklaştırma Cezası:\na) Kavga etmek\nd) Kopya çekmek\ne) Tehdit etmek"
        hit = Hit("yonetmelik.txt", 1, 100.0, text)
        hit.embedding = [0.1]*384
        
        answer = agent.answer("kopya çekmek", [hit])
        
        # d) Kopya çekmek seçilmeli, context'te satırlar bulunmalı
        self.assertIn("Kopya çekmek", answer.finalText)

    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_vector_agent_academic_filtering(self, mock_sim, mock_emb):
        """Akademik filtrelemesini test eder"""
        mock_emb.return_value = [0.1] * 384
        mock_sim.return_value = 0.8
        
        agent = VectorAnswerAgent()
        chunk_text = "Prof. Dr. Mustafa AĞAOĞLU\nOfis: M2-221\nE-posta: m.agaoglu@uni.tr"
        hit = Hit("akademik_kadro.txt", 2, 100.0, chunk_text)
        hit.embedding = [0.1]*384
        
        answer = agent.answer("Prof nerede", [hit])
        
        # Ofis bilgisi gelmeli
        self.assertIn("M2", answer.finalText)

    @patch('src.impl.get_embedding')
    @patch('src.impl.cosine_similarity')
    def test_vector_agent_empty_hits(self, mock_sim, mock_emb):
        """Boş hits ile test eder"""
        mock_emb.return_value = [0.1] * 384
        
        agent = VectorAnswerAgent()
        answer = agent.answer("soru", [])
        
        self.assertEqual(answer.finalText, "Bilgi bulunamadı.")

    # ========================================================================
    # TEST 8: Citation Formatting - Data Integrity
    # ========================================================================
    def test_citation_docid_format(self):
        """Citation docId formatını test eder"""
        agent = KeywordAnswerAgent()
        hit = Hit("belge.txt", 5, 10.0, "İçerik")
        
        answer = agent.answer("soru", [hit])
        
        self.assertEqual(answer.citations[0].docId, "belge.txt")

    def test_citation_section_id_format(self):
        """Citation sectionId formatını test eder"""
        agent = KeywordAnswerAgent()
        hit = Hit("belge.txt", 5, 10.0, "İçerik")
        
        answer = agent.answer("soru", [hit])
        
        self.assertTrue(answer.citations[0].sectionId.startswith("Chunk"))

    def test_multiple_citations(self):
        """Çok sayıda citation test eder"""
        agent = KeywordAnswerAgent()
        hits = [
            Hit("doc1.txt", 0, 100.0, "İçerik 1"),
            Hit("doc2.txt", 1, 90.0, "İçerik 2")
        ]
        
        # Agent en iyi hit'i seçiyor
        answer = agent.answer("soru", hits)
        
        self.assertGreater(len(answer.citations), 0)

    # ========================================================================
    # TEST 9: Hit Sorting & Comparison - Polymorphism
    # ========================================================================
    def test_hit_sorting_by_score_desc(self):
        """Hit'lerin puana göre azalan sırada sort edilmesini test eder"""
        hits = [
            Hit("doc1", 0, 50.0, None),
            Hit("doc2", 0, 100.0, None),
            Hit("doc3", 0, 75.0, None)
        ]
        
        hits.sort()
        
        # En yüksek skor önce gelmelidir
        self.assertEqual(hits[0].score, 100.0)
        self.assertEqual(hits[1].score, 75.0)
        self.assertEqual(hits[2].score, 50.0)

    def test_hit_sorting_by_docid_asc(self):
        """Aynı puan durumunda DocID ASC sıralanmasını test eder"""
        hits = [
            Hit("doc_z", 0, 100.0, None),
            Hit("doc_a", 0, 100.0, None),
            Hit("doc_m", 0, 100.0, None)
        ]
        
        hits.sort()
        
        # Aynı skor'da DocID alfabetik olmalı
        self.assertEqual(hits[0].docId, "doc_a")
        self.assertEqual(hits[1].docId, "doc_m")
        self.assertEqual(hits[2].docId, "doc_z")

    def test_hit_sorting_by_chunkid_asc(self):
        """Aynı DocID'de ChunkID ASC sıralanmasını test eder"""
        hits = [
            Hit("doc", 2, 100.0, None),
            Hit("doc", 0, 100.0, None),
            Hit("doc", 1, 100.0, None)
        ]
        
        hits.sort()
        
        # Aynı DocID'de ChunkID sayısal olmalı
        self.assertEqual(hits[0].chunkId, 0)
        self.assertEqual(hits[1].chunkId, 1)
        self.assertEqual(hits[2].chunkId, 2)

    # ========================================================================
    # TEST 10: Model Data Classes - Encapsulation
    # ========================================================================
    def test_chunk_creation(self):
        """Chunk nesnesi oluşturma test eder"""
        chunk = Chunk(
            docId="test.txt",
            chunkId=0,
            rawText="Test text",
            startOffset=0,
            endOffset=9
        )
        
        self.assertEqual(chunk.docId, "test.txt")
        self.assertEqual(chunk.rawText, "Test text")

    def test_answer_str_representation(self):
        """Answer string representasyonunu test eder"""
        citation = Citation("doc.txt", "Chunk0", 0, 50)
        answer = Answer("Test cevap", [citation])
        
        answer_str = str(answer)
        
        self.assertIn("Test cevap", answer_str)
        self.assertIn("doc.txt", answer_str)

    def test_citation_str_representation(self):
        """Citation string representasyonunu test eder"""
        citation = Citation("doc.txt", "Chunk0", 0, 50)
        citation_str = str(citation)
        
        self.assertIn("doc.txt", citation_str)
        self.assertIn("Chunk0", citation_str)
        self.assertIn("0-50", citation_str)

    # ========================================================================
    # TEST 11: Integration Tests - End-to-End Pipeline
    # ========================================================================
    def test_full_pipeline_course_query(self):
        """Ders sorgusu için tam pipeline test eder"""
        # Setup
        detector = ConfigurableIntentDetector(self.rules)
        writer = HeuristicQueryWriter()
        retriever = KeywordRetriever()
        reranker = SimpleReranker(self.chunks)
        agent = KeywordAnswerAgent()
        
        # Pipeline
        question = "CSE3063 ders kodunun kredisi kaç?"
        intent = detector.detect(question)
        
        self.assertEqual(intent, Intent.COURSE_INFO)
        
        query_terms = writer.write(question, intent)
        self.assertGreater(len(query_terms), 0)
        
        hits = retriever.retrieve(query_terms, self.index)
        self.assertGreater(len(hits), 0)
        
        reranked_hits = reranker.rerank(query_terms, hits)
        self.assertEqual(reranked_hits[0].docId, "ders_planı.txt")
        
        answer = agent.answer(question, reranked_hits)
        self.assertIsNotNone(answer.finalText)
        self.assertGreater(len(answer.citations), 0)

    def test_full_pipeline_staff_query(self):
        """Staff sorgusu için tam pipeline test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        writer = HeuristicQueryWriter()
        retriever = KeywordRetriever()
        
        question = "Mustafa hoca nerede?"
        intent = detector.detect(question)
        
        self.assertEqual(intent, Intent.STAFF_LOOKUP)

    # ========================================================================
    # TEST 12: Edge Cases & Error Handling
    # ========================================================================
    def test_empty_question(self):
        """Boş soru handling test eder"""
        detector = ConfigurableIntentDetector(self.rules)
        result = detector.detect("")
        
        self.assertEqual(result, Intent.UNKNOWN)

    def test_very_long_question(self):
        """Çok uzun soru handling test eder"""
        writer = HeuristicQueryWriter()
        long_question = "test " * 1000
        
        terms = writer.write(long_question, Intent.UNKNOWN)
        
        self.assertGreater(len(terms), 0)

    def test_special_characters_handling(self):
        """Özel karakterler handling test eder"""
        writer = HeuristicQueryWriter()
        question = "CSE@3063!!! ders??? nedir"
        
        terms = writer.write(question, Intent.COURSE_INFO)
        
        # Özel karakterler temizlenmeli
        self.assertTrue(any("3063" in t for t in terms))

    def test_case_insensitive_intent_detection(self):
        """Intent tespiti case-insensitive olmalı"""
        detector = ConfigurableIntentDetector(self.rules)
        
        result1 = detector.detect("DERS KREDISI NEDIR")
        result2 = detector.detect("ders kredisi nedir")
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, Intent.COURSE_INFO)


if __name__ == '__main__':
    unittest.main()