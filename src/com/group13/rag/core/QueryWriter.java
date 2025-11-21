package com.group13.rag.core;

import com.group13.rag.model.Intent;
import java.util.List;

/**
 * [cite_start]Strategy Interface for the Query Writing stage [cite: 103-105].
 * This component is responsible for transforming the raw user question 
 * into a clean list of search terms (tokens) for the Retriever.
 */
public interface QueryWriter {
    /**
     * Rewrites the raw question into effective search terms.
     * It typically handles normalization, stop-word removal, and intent-based boosting.
     */
    List<String> write(String question, Intent intent);
}