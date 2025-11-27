package com.group13.rag.core;

import com.group13.rag.model.Hit;
import java.util.List;

/**
 * [cite_start]Strategy Interface for the Reranking stage [cite: 119-120].
 * This component is responsible for taking the initial set of candidate chunks (Hits)
 * and re-ordering them based on more specific heuristics (e.g., exact match boosting, title proximity).
 */
public interface Reranker {
    // Re-scores and sorts the retrieved hits to improve relevance.
    List<Hit> rerank(List<String> queryTerms, List<Hit> hits);
}