import unittest
from src.models import Intent, KeywordIndex, IndexEntry, Chunk, Hit, Answer
from src.impl import (
    RuleBasedIntentDetector,
    HeuristicQueryWriter,
    KeywordRetriever,
    SimpleReranker,
    TemplateAnswerAgent
)


class RagSystemTest(unittest.TestCase):

    # --- TEST 1: Intent Detection Rules ---
    def test_intent_rules(self):
        detector = RuleBasedIntentDetector()

        # Scenario 1: Staff Lookup
        self.assertEqual(
            Intent.STAFF_LOOKUP,
            detector.detect("Murat hoca nerede?"),
            "Keyword 'hoca' should trigger STAFF intent"
        )
        self.assertEqual(
            Intent.STAFF_LOOKUP,
            detector.detect("Bölüm başkanının ofisi"),
            "Keyword 'ofis' should trigger STAFF intent"
        )

        # Scenario 2: Course Information
        self.assertEqual(
            Intent.COURSE_INFO,
            detector.detect("CSE3063 dersinin önkoşulu nedir?"),
            "Keywords 'Ders/Önkoşul' should trigger COURSE intent"
        )

        # Scenario 3: Policy/Regulations
        self.assertEqual(
            Intent.POLICY_FAQ,
            detector.detect("Mazeret sınavı yönetmeliği"),
            "Keywords 'Yönetmelik/Sınav' should trigger POLICY intent"
        )

    # --- TEST 2: Stopwords & Boosters ---
    def test_stopwords_and_boosters(self):
        writer = HeuristicQueryWriter()

        # Scenario: Stopword Removal
        question = "CSE3063 ve CSE1242 nedir?"
        terms = writer.write(question, Intent.UNKNOWN)

        self.assertIn("cse3063", terms, "Technical terms (CSE3063) must be preserved")
        self.assertNotIn("ve", terms, "Stopword 've' must be removed")
        self.assertNotIn("nedir", terms, "Stopword 'nedir' must be removed")

        # Scenario: Staff Intent Booster
        boosted_terms = writer.write("Murat Can Ganiz", Intent.STAFF_LOOKUP)

        self.assertIn(
            "ofis",
            boosted_terms,
            "The term 'ofis' must be auto-injected when Intent is STAFF_LOOKUP"
        )

    # --- TEST 3: Retrieval Ordering (Coordinate Matching) ---
    def test_retrieval_ordering(self):
        index = KeywordIndex()

        java_entries = [
            IndexEntry(docId="doc1", chunkId=1, tf=1),
            IndexEntry(docId="doc1", chunkId=2, tf=50)
        ]
        oop_entries = [
            IndexEntry(docId="doc1", chunkId=1, tf=1)
        ]

        index.indexMap["java"] = java_entries
        index.indexMap["oop"] = oop_entries

        retriever = KeywordRetriever()
        hits = retriever.retrieve(["java", "oop"], index)

        self.assertEqual(
            1,
            hits[0].chunkId,
            "Chunk with most unique terms must be first"
        )

        self.assertTrue(
            hits[0].score > hits[1].score,
            "Coordinate matching score must outweigh raw TF score"
        )

    # --- TEST 4: Reranker Bonuses ---
    def test_reranker_bonuses(self):
        c1 = Chunk("doc1", 0, "This text contains CSE3063 course.", 0, 0)
        c2 = Chunk("doc1", 1, "This text is irrelevant.", 0, 0)
        chunks = [c1, c2]

        reranker = SimpleReranker(chunks)

        h1 = Hit("doc1", 0, 10.0, None)
        h2 = Hit("doc1", 1, 10.0, None)
        hits = [h1, h2]

        query = ["cse3063"]

        reranker.rerank(query, hits)

        self.assertTrue(
            hits[0].score > 10.0,
            "Chunk containing exact course code must receive score boost"
        )
        self.assertEqual(
            0,
            hits[0].chunkId,
            "Relevant chunk must be ranked first"
        )

    # --- TEST 5: Sentence Selection & Citation Formatting ---
    def test_sentence_selection_and_citation(self):
        agent = TemplateAnswerAgent()

        text = (
            "Dönem: Güz\n"
            "CSE3055 Database Systems\n"
            "CSE3063 Object Oriented Design - Önkoşul: CSE1242\n"
            "CSE3215 Digital Logic"
        )

        hit = Hit("ders_plani.txt", 5, 100.0, text)
        hits = [hit]

        answer = agent.answer("CSE3063 önkoşulu nedir?", hits)

        final_text = answer.finalText

        self.assertIn("CSE3063", final_text, "Answer must contain the specific course line")
        self.assertIn("CSE1242", final_text, "Answer must contain the prerequisite info")

        self.assertTrue(len(answer.citations) > 0, "Citation list must not be empty")
        self.assertEqual("ders_plani.txt", answer.citations[0].docId)

        self.assertIn(
            "ders_plani.txt:P5",
            str(answer),
            "Citation string format must be correct"
        )


if __name__ == "__main__":
    unittest.main()
