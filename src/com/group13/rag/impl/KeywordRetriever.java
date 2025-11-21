package com.group13.rag.impl;

import com.group13.rag.core.Retriever;
import com.group13.rag.model.Hit;
import com.group13.rag.model.IndexEntry;
import com.group13.rag.model.KeywordIndex;

import java.util.*;

/**
 * [cite_start]Implementation of the Retrieval Strategy [cite: 125-129].
 * Searches the inverted index to find candidate documents.
 */
public class KeywordRetriever implements Retriever {

    @Override
    public List<Hit> retrieve(List<String> queryTerms, KeywordIndex index) {
        // Map to store total TF (Term Frequency) score per Chunk
        Map<String, Double> scoreMap = new HashMap<>();
        
        // Map to track UNIQUE term matches (Coordinate Matching)
        Map<String, Set<String>> matchCounts = new HashMap<>();

        // 1. Inverted Index Lookup (Term -> Entries)
        for (String term : queryTerms) {
            if (index.getIndexMap().containsKey(term)) {
                List<IndexEntry> entries = index.getIndexMap().get(term);
                
                for (IndexEntry entry : entries) {
                    // Unique Key construction: docId + "::" + chunkId
                    String key = entry.getDocId() + "::" + entry.getChunkId();
                    
                    // A. Accumulate TF Score (Baseline Logic) [cite: 127]
                    double currentScore = scoreMap.getOrDefault(key, 0.0);
                    scoreMap.put(key, currentScore + entry.getTf());
                    
                    // B. Track Unique Matches (Optimization)
                    matchCounts.putIfAbsent(key, new HashSet<>());
                    matchCounts.get(key).add(term);
                }
            }
        }
        
        // 2. Convert Maps to Hit List
        List<Hit> hits = new ArrayList<>();
        for (Map.Entry<String, Double> entry : scoreMap.entrySet()) {
            String key = entry.getKey();
            String[] parts = key.split("::");
            String docId = parts[0];
            int chunkId = Integer.parseInt(parts[1]);
            
            double tfScore = entry.getValue();
            int distinctMatches = matchCounts.get(key).size();
            
            // SCORING FORMULA: (Unique Match Count * 1000) + TF Score
            // This ensures a document containing "CSE3063" AND "Prerequisite" (Score ~2000)
            // ranks higher than a document with "CSE3063" appearing 50 times (Score ~50).
            double finalScore = (distinctMatches * 1000.0) + tfScore;
            
            hits.add(new Hit(docId, chunkId, finalScore, null));
        }
        
        // 3. Sort Results (Deterministic Tie-Break via Hit.compareTo) [cite: 130]
        Collections.sort(hits);
        
        // Return all candidates (Filtering happens in Reranker)
        return hits;
    }
}