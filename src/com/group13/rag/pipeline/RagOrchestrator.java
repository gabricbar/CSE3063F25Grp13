package com.group13.rag.pipeline;

import com.group13.rag.core.*;
import com.group13.rag.model.*;
import com.group13.rag.tracing.TraceBus;

import java.util.List;

// GRASP Controller: Coordinates the RAG pipeline workflow [cite: 83]
public class RagOrchestrator {
    
    // Strategy Interfaces (Dependencies)
    private final IntentDetector intentDetector;
    private final QueryWriter queryWriter;
    private final Retriever retriever;
    private final Reranker reranker;
    private final AnswerAgent answerAgent;
    
    // Data Source
    private final KeywordIndex globalIndex;

    // Constructor Injection: Applies Dependency Inversion Principle (DIP) [cite: 55, 178]
    public RagOrchestrator(IntentDetector id, QueryWriter qw, Retriever ret, 
                           Reranker re, AnswerAgent aa, KeywordIndex index) {
        this.intentDetector = id;
        this.queryWriter = qw;
        this.retriever = ret;
        this.reranker = re;
        this.answerAgent = aa;
        this.globalIndex = index;
    }

    // Sequential Pipeline Execution (Template Method) [cite: 145-148]
    public String run(String userQuestion) {
        long startTime = System.currentTimeMillis();
        TraceBus.push("START", "Question received: " + userQuestion);

        // 1. Intent Detection: Classify the question type (Staff, Course, etc.)
        Intent intent = intentDetector.detect(userQuestion);
        TraceBus.push("INTENT", "Detected intent: " + intent);

        // 2. Query Writing: Extract keywords and normalize input
        List<String> terms = queryWriter.write(userQuestion, intent);
        TraceBus.push("QUERY", "Search terms: " + terms);

        // 3. Retrieval: Search the inverted index for candidates
        List<Hit> hits = retriever.retrieve(terms, globalIndex);
        TraceBus.push("RETRIEVE", "Found chunks: " + hits.size());

        // 4. Reranking: Optimize the order of results (e.g., prioritize exact codes)
        List<Hit> rerankedHits = reranker.rerank(terms, hits);
        if (!rerankedHits.isEmpty()) {
            TraceBus.push("RERANK", "Best score: " + rerankedHits.get(0).score);
        }

        // 5. Answer Generation: Select best text and format citations
        Answer finalAnswer = answerAgent.answer(userQuestion, rerankedHits);
        TraceBus.push("ANSWER", "Answer generated.");

        // Performance Logging
        long duration = System.currentTimeMillis() - startTime;
        TraceBus.push("END", "Total duration: " + duration + "ms");

        return finalAnswer.toString();
    }
}