package com.group13.rag.core;

import com.group13.rag.model.Answer;
import com.group13.rag.model.Hit;
import java.util.List;

/**
 * Strategy Interface for the Answer Generation stage [cite: 136-138].
 * This component is responsible for synthesizing the final response presented to the user
 * based on the top-ranked retrieval results.
 */
public interface AnswerAgent {
    //Constructs the final answer string and associated citations.
    Answer answer(String question, List<Hit> topHits);
}