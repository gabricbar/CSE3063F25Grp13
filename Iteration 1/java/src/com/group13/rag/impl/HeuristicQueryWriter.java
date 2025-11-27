package com.group13.rag.impl;

import com.group13.rag.core.QueryWriter;
import com.group13.rag.model.Intent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * [cite_start]Implementation of the Query Writing Strategy [cite: 103-109].
 * This class transforms the raw user question into a clean list of search terms.
 * It performs normalization, stop-word removal, and intent-based boosting.
 */
public class HeuristicQueryWriter implements QueryWriter {

    // Extended Stop Words List to reduce "Query Noise"
    // Removing these common words improves retrieval accuracy significantly.
    private static final List<String> STOP_WORDS = Arrays.asList(
        // Question particles
        "nedir", "kimdir", "nasıl", "nerede", "hangi", "kaç", "mi", "mı", "mu", "mü", "soru",
        // Conjunctions and pronouns
        "ve", "ile", "için", "bu", "şu", "o", "bir", "var", "yok", "veya", "olarak",
        // Domain-specific noise (words that appear in almost every document)
        "ders", "dersi", "dersinin", "hakkında", "bilgi", "ilgili", "kısmı", "bölüm", "mühendisliği"
    );

    @Override
    public List<String> write(String question, Intent intent) {
        List<String> terms = new ArrayList<>();
        
        // 1. Normalization
        // - Convert to Lowercase using Turkish Locale to handle 'I-ı' and 'İ-i' correctly.
        // - Replace all non-alphanumeric characters with space using Unicode Regex.
        // - [^\\p{L}0-9] means: "Any character that is NOT a Letter (in any language) or a Number".
        String normalized = question.toLowerCase(new Locale("tr", "TR"))
                                    .replaceAll("[^\\p{L}0-9]", " ");
        
        // 2. Tokenization: Split by whitespace
        String[] tokens = normalized.split("\\s+");
        
        for (String token : tokens) {
            // Skip very short tokens (length < 2)
            if (token.length() < 2) continue; 
            
            // 3. Filtering
            if (!STOP_WORDS.contains(token) || token.matches(".*\\d.*")) {
                terms.add(token);
            }
        }
        
        // 4. Intent Boosting
        if (intent == Intent.STAFF_LOOKUP) {
            if (!terms.contains("ofis")) terms.add("ofis");
        }
        
        return terms;
    }
}