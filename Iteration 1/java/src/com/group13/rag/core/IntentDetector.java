package com.group13.rag.core;

import com.group13.rag.model.Intent;

/**
 * [cite_start]Strategy Interface for the Intent Detection stage [cite: 106-108].
 * This component is responsible for analyzing the user's raw input question 
 * and classifying it into a specific domain intent (e.g., Staff Lookup, Course Info).
 */
public interface IntentDetector {
    // Analyzes the input question to determine the underlying user intent.
    Intent detect(String question);
}