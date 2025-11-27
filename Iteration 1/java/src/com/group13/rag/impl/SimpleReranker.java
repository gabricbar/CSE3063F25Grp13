package com.group13.rag.impl;

import com.group13.rag.core.Reranker;
import com.group13.rag.model.Chunk;
import com.group13.rag.model.Hit;

import java.util.*;
import java.util.stream.Collectors;

public class SimpleReranker implements Reranker {

    private final Map<String, Chunk> chunkMap;

    public SimpleReranker(List<Chunk> allChunks) {
        this.chunkMap = allChunks.stream()
            .collect(Collectors.toMap(
                c -> c.getDocId() + "_" + c.getChunkId(),
                c -> c,
                (a, b) -> a
            ));
    }

    @Override
    public List<Hit> rerank(List<String> queryTerms, List<Hit> hits) {

        for (Hit hit : hits) {

            String key = hit.docId + "_" + hit.chunkId;
            Chunk chunk = chunkMap.get(key);
            if (chunk == null) continue;

            hit.chunkText = chunk.getRawText();
            String text = hit.chunkText.toLowerCase(Locale.ROOT);

            // -------- 1. tf_sum yeniden hesaplanıyor --------
            int tfSum = calculateTfSum(text, queryTerms);

            // -------- 2. proximity bonus --------
            int proximityBonus = calculateProximityBonus(text, queryTerms);

            // -------- 3. title boost --------
            int titleBoost = calculateTitleBoost(hit.docId, queryTerms);

            // -------- 4. Final Score --------
            hit.score = (tfSum * 10)
                        + proximityBonus
                        + titleBoost;
        }

        Collections.sort(hits); // stable

        return hits;
    }


    // ========== HELPERS ==========

    private int calculateTfSum(String text, List<String> terms) {
        int sum = 0;
        for (String term : terms) {
            int index = 0;
            String t = term.toLowerCase(Locale.ROOT);
            while ((index = text.indexOf(t, index)) != -1) {
                sum++;
                index += t.length();
            }
        }
        return sum;
    }

    private int calculateProximityBonus(String text, List<String> terms) {
        List<Integer> positions = new ArrayList<>();

        for (String term : terms) {
            int index = text.indexOf(term.toLowerCase(Locale.ROOT));
            if (index != -1) positions.add(index);
        }

        if (positions.size() < 2) return 0;

        Collections.sort(positions);

        for (int i = 0; i < positions.size() - 1; i++) {
            if (Math.abs(positions.get(i) - positions.get(i + 1)) <= 15) {
                return 5;
            }
        }

        return 0;
    }

    private int calculateTitleBoost(String docId, List<String> terms) {
        String title = docId.toLowerCase(Locale.ROOT);
        for (String t : terms) {
            if (title.contains(t.toLowerCase(Locale.ROOT))) {
                return 3;
            }
        }
        return 0;
    }
}