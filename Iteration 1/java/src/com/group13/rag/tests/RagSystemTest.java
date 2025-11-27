package com.group13.rag.tests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

// Import core implementation and model classes
import com.group13.rag.impl.*;
import com.group13.rag.model.*;
import java.util.*;

/**
 * Comprehensive Unit Test Suite for the RAG Pipeline.
 * Covers requirements: Intent Rules, Stopwords, Retrieval Ordering, Reranker Bonuses, and Citation Formatting.
 */
class RagSystemTest {

    // --- TEST 1: Intent Detection Rules ---
    @Test
    void testIntentRules() {
        RuleBasedIntentDetector detector = new RuleBasedIntentDetector();
        
        // Scenario 1: Staff Lookup
        assertEquals(Intent.STAFF_LOOKUP, detector.detect("Murat hoca nerede?"), 
            "Keyword 'hoca' should trigger STAFF intent");
        assertEquals(Intent.STAFF_LOOKUP, detector.detect("Bölüm başkanının ofisi"), 
            "Keyword 'ofis' should trigger STAFF intent");
        
        // Scenario 2: Course Information
        assertEquals(Intent.COURSE_INFO, detector.detect("CSE3063 dersinin önkoşulu nedir?"), 
            "Keywords 'Ders/Önkoşul' should trigger COURSE intent");
        
        // Scenario 3: Policy/Regulations
        assertEquals(Intent.POLICY_FAQ, detector.detect("Mazeret sınavı yönetmeliği"), 
            "Keywords 'Yönetmelik/Sınav' should trigger POLICY intent");
    }

    // --- TEST 2: Stopwords & Boosters ---
    @Test
    void testStopwordsAndBoosters() {
        HeuristicQueryWriter writer = new HeuristicQueryWriter();
        
        // Scenario: Stopword Removal
        String question = "CSE3063 ve CSE1242 nedir?"; // 've', 'nedir' are stopwords
        List<String> terms = writer.write(question, Intent.UNKNOWN);
        
        assertTrue(terms.contains("cse3063"), "Technical terms (CSE3063) must be preserved");
        assertFalse(terms.contains("ve"), "Stopword 've' must be removed");
        assertFalse(terms.contains("nedir"), "Stopword 'nedir' must be removed");

        // Scenario: Staff Intent Booster
        String staffQuestion = "Murat Can Ganiz";
        List<String> boostedTerms = writer.write(staffQuestion, Intent.STAFF_LOOKUP);
        
        assertTrue(boostedTerms.contains("ofis"), 
            "The term 'ofis' must be auto-injected when Intent is STAFF_LOOKUP");
    }

    // --- TEST 3: Retrieval Ordering (Coordinate Matching) ---
    @Test
    void testRetrievalOrdering() {
        // Setup a Mock Index
        KeywordIndex index = new KeywordIndex();
        Map<String, List<IndexEntry>> map = new HashMap<>();
        
        // Scenario: 
        // Chunk 1: Contains both "java" and "oop" (2 unique matches) -> Should rank higher
        // Chunk 2: Contains only "java" many times (1 unique match) -> Should rank lower
        
        List<IndexEntry> javaEntries = Arrays.asList(
            new IndexEntry("doc1", 1, 1), // Exists in Chunk 1
            new IndexEntry("doc1", 2, 50) // Exists in Chunk 2 (High TF)
        );
        List<IndexEntry> oopEntries = Arrays.asList(
            new IndexEntry("doc1", 1, 1)  // Exists in Chunk 1
        );
        
        map.put("java", javaEntries);
        map.put("oop", oopEntries);
        index.setIndexMap(map);
        
        KeywordRetriever retriever = new KeywordRetriever();
        List<Hit> hits = retriever.retrieve(Arrays.asList("java", "oop"), index);
        
        // Verification
        assertEquals(1, hits.get(0).chunkId, "Chunk with most unique terms must be first");
        assertTrue(hits.get(0).score > hits.get(1).score, "Coordinate matching score must outweigh raw TF score");
    }

    // --- TEST 4: Reranker Bonuses ---
    @Test
    void testRerankerBonuses() {
        // Setup Mock Data
        Chunk c1 = new Chunk("doc1", 0, "This text contains CSE3063 course.", 0, 0);
        Chunk c2 = new Chunk("doc1", 1, "This text is irrelevant.", 0, 0);
        List<Chunk> chunks = Arrays.asList(c1, c2);
        
        SimpleReranker reranker = new SimpleReranker(chunks);
        
        // Initial Hits (Equal scores)
        Hit h1 = new Hit("doc1", 0, 10.0, null);
        Hit h2 = new Hit("doc1", 1, 10.0, null);
        List<Hit> hits = new ArrayList<>(Arrays.asList(h1, h2));
        
        // Query contains exact course code
        List<String> query = Arrays.asList("cse3063");
        
        reranker.rerank(query, hits);
        
        // Verification
        assertTrue(h1.score > 200.0, "Chunk containing exact course code must receive 'Silver Bullet' bonus");
    }

    // --- TEST 5: Sentence Selection & Citation Formatting ---
    @Test
    void testSentenceSelectionAndCitation() {
        TemplateAnswerAgent agent = new TemplateAnswerAgent();
        
        // Mock Chunk Text (Simulating a Course Syllabus)
        String text = "Dönem: Güz\n" +
                      "CSE3055 Database Systems\n" +
                      "CSE3063 Object Oriented Design - Önkoşul: CSE1242\n" +
                      "CSE3215 Digital Logic";
        
        // Setup Hit
        Hit hit = new Hit("ders_plani.txt", 5, 100.0, text);
        List<Hit> hits = Collections.singletonList(hit);
        
        // Question: "CSE3063 Prerequisite"
        Answer answer = agent.answer("CSE3063 önkoşulu nedir?", hits);
        
        // Check 1: Sentence Selection (Must extract the specific course line)
        String finalText = answer.getFinalText();
        assertTrue(finalText.contains("CSE3063"), "Answer must contain the specific course line");
        assertTrue(finalText.contains("CSE1242"), "Answer must contain the prerequisite info");
        
        // Check 2: Citation Formatting
        assertFalse(answer.getCitations().isEmpty(), "Citation list must not be empty");
        assertEquals("ders_plani.txt", answer.getCitations().get(0).getDocId());
        
        // Verify formatting in toString() (Expected: P5 for Chunk 5)
        assertTrue(answer.toString().contains("ders_plani.txt:P5"), "Citation string format must be correct");
    }
}
