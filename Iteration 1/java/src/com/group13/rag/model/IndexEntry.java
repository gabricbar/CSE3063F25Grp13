package com.group13.rag.model;

/*
  Represents a single entry (posting) in the Keyword Index.
  Maps a specific term to a specific location (Document + Chunk) and stores the Term Frequency (TF).
  This structure corresponds to the tuple [docId, chunkId, tf] required for the retrieval phase.
 */
public class IndexEntry {
    
    private String docId;   // Identifier of the document containing the term
    private int chunkId;    // Identifier of the specific chunk within that document
    private int tf;         // Term Frequency: How many times the term appears in this chunk

    /*
      Constructs a new IndexEntry.
      @param docId   The document identifier.
      @param chunkId The chunk identifier.
      @param tf      The term frequency count.
     */
    public IndexEntry(String docId, int chunkId, int tf) {
        this.docId = docId;
        this.chunkId = chunkId;
        this.tf = tf;
    }

    // Getters
    public String getDocId() { return docId; }
    public int getChunkId() { return chunkId; }
    public int getTf() { return tf; }
}