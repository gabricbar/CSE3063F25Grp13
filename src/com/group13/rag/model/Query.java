package com.group13.rag.model;

/*
  Represents the user's input query within the system.
  Corresponds to the 'Query' entity in the Domain Model.
  Encapsulates the raw question text and its detected intent.
 */
public class Query {
    
    // The original, unprocessed text of the question asked by the user. 
    private String rawQuestion; 
    
    // The classified intent behind the query (e.g., COURSE_INFO). Represents the "Has 1 Intent" relationship. 
    private Intent intent;      

    /*
      Constructs a new Query object with the provided text.
      The intent is initialized to UNKNOWN by default until processed by the IntentDetector.
      @param rawQuestion The user's input string.
     */
    public Query(String rawQuestion) {
        this.rawQuestion = rawQuestion;
        this.intent = Intent.UNKNOWN; 
    }

    // Getters and Setters

    public String getRawQuestion() { return rawQuestion; }
    
    public Intent getIntent() { return intent; }
    
    public void setIntent(Intent intent) { this.intent = intent; }
}