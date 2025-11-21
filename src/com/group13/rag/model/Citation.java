package com.group13.rag.model;

/*
  Represents a reference to a specific part of a source document used to support an answer.
  Corresponds to the 'Citation' entity in the Domain Model.
  Ensures traceability from the generated answer back to the source text.
 */
public class Citation {
    
    private String docId;      // Identifier of the source document
    private String sectionId;  // Specific section or header within the document (optional)
    private int startOffset;   // Starting character index of the cited text segment
    private int endOffset;     // Ending character index of the cited text segment

    //Constructs a new Citation instance with location details.
     
    public Citation(String docId, String sectionId, int start, int end) {
        this.docId = docId;
        this.sectionId = sectionId;
        this.startOffset = start;
        this.endOffset = end;
    }

    // Getters
    public String getDocId() { return docId; }
    public int getStartOffset() { return startOffset; }
    public int getEndOffset() { return endOffset; }

    /*
      Returns the string representation of the citation in the required format.
      Format: docid:section:start-end
      Example: "regulations.txt:Section5:100-150"
     */
    @Override
    public String toString() {
        return docId + ":" + (sectionId != null ? sectionId : "General") + ":" + startOffset + "-" + endOffset;
    }
}