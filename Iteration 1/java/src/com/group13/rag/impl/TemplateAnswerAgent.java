package com.group13.rag.impl;

import com.group13.rag.core.AnswerAgent;
import com.group13.rag.model.Answer;
import com.group13.rag.model.Citation;
import com.group13.rag.model.Hit;

import java.util.*;

/**
 * Implementation of the Answer Generation Strategy.
 * Uses a "Hybrid Extraction" logic to handle different types of content:
 * 1. Course Info: Extracts specific lines + headers (e.g., "Semester").
 * 2. Staff/General Info: Extracts complete text blocks based on delimiters.
 */
public class TemplateAnswerAgent implements AnswerAgent {

    @Override
    public Answer answer(String question, List<Hit> topHits) {
        // 1. Fallback: Return default message if no hits found
        if (topHits == null || topHits.isEmpty()) {
            return new Answer("Sorry, I could not find any information regarding this.", new ArrayList<>());
        }

        // 2. Pre-processing: Get the best chunk and normalize text
        Hit bestHit = topHits.get(0);
        String rawText = bestHit.chunkText != null ? bestHit.chunkText.replace("\\n", "\n") : "";
        
        String[] lines = rawText.split("\n");
        String[] queryTerms = question.toLowerCase(new Locale("tr", "TR"))
                                      .replaceAll("[^\\p{L}0-9 ]", "")
                                      .split("\\s+");

        // Check if the query contains digits (likely a Course Code like CSE3063)
        boolean hasDigit = false;
        for (String t : queryTerms) if (t.matches(".*\\d+.*")) hasDigit = true;

        List<Integer> targetIndices = new ArrayList<>();

        // 3. Line Selection Strategy
        if (hasDigit) {
            // Look for specific lines containing the digits.
            for (int i = 0; i < lines.length; i++) {
                String lower = lines[i].toLowerCase(new Locale("tr", "TR"));
                for (String t : queryTerms) {
                    if (t.matches(".*\\d+.*") && lower.contains(t)) {
                        targetIndices.add(i);
                    }
                }
            }
        } else {
            // Find the line with the highest keyword density.
            int bestIdx = -1, maxScore = -1;
            for (int i = 0; i < lines.length; i++) {
                int score = 0;
                String lower = lines[i].toLowerCase(new Locale("tr", "TR"));
                for (String t : queryTerms) {
                    if (t.length() > 2 && lower.contains(t)) score++;
                }
                if (score > maxScore) {
                    maxScore = score;
                    bestIdx = i;
                }
            }
            if (bestIdx != -1) targetIndices.add(bestIdx);
        }

        // 4. Context Formatting
        StringBuilder finalOutput = new StringBuilder();
        Set<Integer> processedLines = new HashSet<>();

        for (int idx : targetIndices) {
            String line = lines[idx].trim();
            if (line.isEmpty()) continue;

            // Regex to identify if the line is a Course Entry (e.g., "CSE3063...")
            boolean isCourseLine = line.matches("^\\s*[A-Z]{2,}\\s*[0-9]{3,}.*");

            if (isCourseLine) {
                // Look back up to 6 lines to find the "Semester/Term" header
                for (int k = idx - 1; k >= Math.max(0, idx - 6); k--) {
                    String prev = lines[k].toLowerCase(new Locale("tr", "TR"));
                    if (prev.contains("dönem") || prev.contains("semester") || prev.contains("yarıyıl")) {
                        if (!processedLines.contains(k)) {
                            finalOutput.append(lines[k].trim()).append("\n");
                            processedLines.add(k);
                        }
                        break;
                    }
                }
                // Add the course line itself
                if (!processedLines.contains(idx)) {
                    finalOutput.append(line).append("\n");
                    processedLines.add(idx);
                }
            } else {
                // Expand context up and down until a delimiter (empty line or Title) is found
                int start = idx, end = idx;
                
                // Expand Up
                while (start > 0) {
                    String p = lines[start - 1].trim();
                    if (p.isEmpty() || isTitleLine(p)) break; 
                    if (isTitleLine(p) && start-1 != idx) break;
                    start--;
                }
                // Expand Down
                while (end < lines.length - 1) {
                    String n = lines[end + 1].trim();
                    if (n.isEmpty() || isTitleLine(n)) break;
                    end++;
                }
                // Append the block
                for (int k = start; k <= end; k++) {
                    if (!processedLines.contains(k)) {
                        finalOutput.append(lines[k].trim()).append("\n");
                        processedLines.add(k);
                    }
                }
            }
        }

        // 5. Finalize Text
        String finalText = finalOutput.toString().trim();
        // Fallback: If extraction failed, use raw text truncated to 300 chars
        if (finalText.isEmpty()) finalText = rawText.length() > 300 ? rawText.substring(0, 300) + "..." : rawText;

        // 6. Create Citation
        List<Citation> citations = new ArrayList<>();
        citations.add(new Citation(bestHit.docId, "P" + bestHit.chunkId, 0, 0));
        
        // Return Domain Object
        return new Answer(finalText, citations);
    }

    /**
     * Helper: Checks if a line starts with an academic title.
     */
    private boolean isTitleLine(String line) {
        String l = line.toLowerCase(new Locale("tr", "TR"));
        return l.startsWith("prof") || l.startsWith("doç") || l.startsWith("dr") || l.startsWith("öğr") || l.startsWith("arş");
    }
}