package com.group13.rag.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
  Represents the inverted index (catalog) used for keyword-based retrieval.
  Maps unique tokens (words) to a list of their occurrences across the document corpus.
  Corresponds to the 'KeywordIndex' entity in the Domain Model.
 */
public class KeywordIndex {
    
    /*
      The core data structure mapping a Token (String) to a List of IndexEntry objects.
      Implements the structure requirement: Token -> List of (docId, chunkId, tf).
     */
    private Map<String, List<IndexEntry>> indexMap;

    
     //Constructs a new, empty KeywordIndex.
     
    public KeywordIndex() {
        this.indexMap = new HashMap<>();
    }

    // Getters and Setters

    public Map<String, List<IndexEntry>> getIndexMap() {
        return indexMap;
    }

    public void setIndexMap(Map<String, List<IndexEntry>> indexMap) {
        this.indexMap = indexMap;
    }
}