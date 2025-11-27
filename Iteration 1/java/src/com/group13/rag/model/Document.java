package com.group13.rag.model;

import java.util.List;

/*
  Represents a source document (e.g., a PDF or text file) ingested into the system.
  Corresponds to the 'Document' entity in the Domain Model.
  Acts as a container for multiple Chunks.
 */
public class Document {
    
    private String docId;       // Unique identifier (e.g., filename like 'regulations.txt')
    private String title;       // Title or display name of the document
    private String sourcePath;  // File system path to the original source file
    
    // Represents the "Is composed of" relationship (1 Document -> Many Chunks) defined in the Domain Model
    private List<Chunk> chunks; 

    // Constructs a new Document with metadata.
    public Document(String docId, String title, String sourcePath) {
        this.docId = docId;
        this.title = title;
        this.sourcePath = sourcePath;
    }

    // Getters and Setters
    public String getDocId() { return docId; }
    public String getTitle() { return title; }
    public String getSourcePath() { return sourcePath; }

    public List<Chunk> getChunks() { return chunks; }
    public void setChunks(List<Chunk> chunks) { this.chunks = chunks; }
}