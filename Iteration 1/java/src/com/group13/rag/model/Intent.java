package com.group13.rag.model;

/*
  Represents the categorized intention behind a user's question.
  Used by the IntentDetector strategy to route the query to specific logic.
  Corresponds to the classifications defined in the project requirements.
 */
public enum Intent {
    
    // Queries related to course registration, enrollment, or freezing registration. 
    REGISTRATION,   
    
    // Queries regarding faculty members, staff contact information, or office locations. 
    STAFF_LOOKUP,   
    
    // General questions about university regulations, internships, exams, and policies. 
    POLICY_FAQ,     
    
    //Specific inquiries about course details, credits (ECTS), prerequisites, or schedules. 
    COURSE_INFO,    
    
    // Fallback category when the system cannot confidently classify the user's intent.
    UNKNOWN         
}