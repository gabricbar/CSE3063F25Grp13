package com.group13.rag.model;

/*
  Represents a specific text segment (chunk) derived from a source document.
  This is the atomic unit of retrieval in the RAG system.
  Corresponds to the 'Chunk' entity in the Domain Model.
 */
public class Chunk {
    
    private String docId;      // Identifier of the parent document
    private int chunkId;       // Unique sequence identifier for this chunk within the document
    private String rawText;    // The actual text content of the chunk
    private String sectionId;  // Optional: The section or header this chunk belongs to
    private int startOffset;   // Character index where this chunk starts in the original text
    private int endOffset;     // Character index where this chunk ends

    // No-args constructor required for JSON deserialization (Gson).
     
    public Chunk() {
    }

    // Constructs a new Chunk with specific metadata.
     
    public Chunk(String docId, int chunkId, String rawText, int start, int end) {
        this.docId = docId;
        this.chunkId = chunkId;
        this.rawText = rawText;
        this.startOffset = start;
        this.endOffset = end;
    }

    // Getters and Setters

    public String getDocId() { return docId; }
    
    public int getChunkId() { return chunkId; }
    
    public String getRawText() { return rawText; }
    
    public int getStartOffset() { return startOffset; }
    
    public int getEndOffset() { return endOffset; }
    
    public void setSectionId(String sectionId) { this.sectionId = sectionId; }
    public String getSectionId() { return sectionId; }
}