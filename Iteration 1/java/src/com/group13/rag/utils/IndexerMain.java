package com.group13.rag.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.group13.rag.model.Chunk;
import com.group13.rag.model.IndexEntry;
import com.group13.rag.model.KeywordIndex;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class to ingest text documents, split them into chunks (paragraphs),
 * and build a simple keyword inverted index.
 */
public class IndexerMain {

    public static void main(String[] args) {
        System.out.println("=== STARTING PARAGRAPH-BASED INDEXING ===");

        // 1. Check Input Directory
        File corpusDir = new File("data/corpus");
        if (!corpusDir.exists()) {
            System.err.println("ERROR: 'data/corpus' directory not found!");
            return;
        }

        // 2. Initialize Data Structures
        List<Chunk> allChunks = new ArrayList<>();
        Map<String, List<IndexEntry>> rawIndexMap = new HashMap<>();

        // 3. Process Files
        File[] files = corpusDir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.getName().toLowerCase().endsWith(".txt")) {
                    System.out.println("Processing: " + file.getName());
                    processFileByParagraphs(file, allChunks, rawIndexMap);
                }
            }
        }

        // 4. Setup Gson for JSON Serialization (Pretty Printing enabled for readability)
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        // 5. Save Chunks to JSON
        try (Writer writer = new FileWriter("data/chunks.json")) {
            gson.toJson(allChunks, writer);
            System.out.println(">> 'data/chunks.json' created. Total Chunks: " + allChunks.size());
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 6. Wrap and Save Index to JSON
        KeywordIndex keywordIndex = new KeywordIndex();
        keywordIndex.setIndexMap(rawIndexMap);

        try (Writer writer = new FileWriter("data/index.json")) {
            gson.toJson(keywordIndex, writer);
            System.out.println(">> 'data/index.json' created.");
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        System.out.println("=== INDEXING COMPLETED ===");
    }

    /**
     * Reads a file, splits it into paragraphs, creates Chunks, and updates the index.
     */
    private static void processFileByParagraphs(File file, List<Chunk> allChunks, Map<String, List<IndexEntry>> index) {
        try {
            String content = Files.readString(file.toPath());
            String docId = file.getName();

            //"\\r?\\n){2,}" matches double newlines.
            // This Regex splits the text into paragraphs based on empty lines.
            String[] paragraphs = content.split("(\\r?\\n){2,}");

            int chunkId = 0;
            for (String para : paragraphs) {
                String trimmed = para.trim();
                
                // Skip very short paragraphs or empty strings
                if (trimmed.length() < 5) continue; 

                // Create a single chunk object
                Chunk chunk = new Chunk(docId, chunkId++, trimmed, 0, 0);
                allChunks.add(chunk);
                
                // Update the inverted index with tokens from this chunk
                updateIndex(index, trimmed, docId, chunk.getChunkId());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Tokenizes the text and updates the global inverted index map.
     */
    private static void updateIndex(Map<String, List<IndexEntry>> index, String text, String docId, int chunkId) {
        // Tokenize: Split by non-alphanumeric characters (supports Turkish characters)
        String[] tokens = text.toLowerCase().split("[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ]+");
        
        // Calculate Local Term Frequency (TF) for this specific chunk
        Map<String, Integer> localTf = new HashMap<>();
        for (String token : tokens) {
            // Filter out short tokens (length < 3)
            if (token.length() < 3) continue; 
            localTf.put(token, localTf.getOrDefault(token, 0) + 1);
        }
        
        // Update the global index
        for (Map.Entry<String, Integer> entry : localTf.entrySet()) {
            index.putIfAbsent(entry.getKey(), new ArrayList<>());
            
            // Add the new entry (Location + TF)
            index.get(entry.getKey()).add(new IndexEntry(docId, chunkId, entry.getValue()));
        }
    }
}