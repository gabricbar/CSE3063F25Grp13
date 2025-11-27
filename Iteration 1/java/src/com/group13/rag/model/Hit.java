package com.group13.rag.model;

/*
  Represents a candidate search result (a "hit") returned by the Retriever.
  Implements Comparable to support deterministic sorting based on score and metadata.
  This class is essential for the Retrieval and Reranking stages.
 */
public class Hit implements Comparable<Hit> {
    
    public String docId;       // Identifier of the source document
    public int chunkId;        // Unique identifier of the chunk within the document
    public double score;       // Relevance score assigned by Retriever or Reranker
    public String chunkText;   // The text content of the chunk (carried for display purposes)

    //Constructs a new Hit instance.
     
    public Hit(String docId, int chunkId, double score, String text) {
        this.docId = docId;
        this.chunkId = chunkId;
        this.score = score;
        this.chunkText = text;
    }

    /*
      Implements the "Deterministic Tie-Break" rule required by the project specification.
      Sorting Order:
      1. Score DESC (Higher score comes first)
      2. DocID ASC (Alphabetical order for ties)
      3. ChunkID ASC (Earlier chunks for ties)
     */
    @Override
    public int compareTo(Hit other) {
        // 1. Compare Score (DESC)
        int scoreCmp = Double.compare(other.score, this.score);
        if (scoreCmp != 0) return scoreCmp;
        
        // 2. Compare DocID (ASC) - Tie-breaker 1
        int docCmp = this.docId.compareTo(other.docId);
        if (docCmp != 0) return docCmp;
        
        // 3. Compare ChunkID (ASC) - Tie-breaker 2
        return Integer.compare(this.chunkId, other.chunkId);
    }
}