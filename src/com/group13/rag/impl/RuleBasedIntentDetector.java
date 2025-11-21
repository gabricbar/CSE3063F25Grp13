package com.group13.rag.impl;

import com.group13.rag.core.IntentDetector;
import com.group13.rag.model.Intent;

/**
 * Baseline implementation of the Intent Detection Strategy.
 * Uses simple keyword matching rules to classify the user's question.
 * * Reference: Document Item 110-113
 */
public class RuleBasedIntentDetector implements IntentDetector {

    @Override
    public Intent detect(String question) {
        String q = question.toLowerCase();

        // Priority Order: Staff > Course > Policy > Registration (Deterministic tie-breaking)
        
        // 1. Staff / Personnel Lookup
        if (q.contains("hoca") || q.contains("ofis") || q.contains("mail") || 
            q.contains("iletişim") || q.contains("kimdir") || q.contains("başkan")) {
            return Intent.STAFF_LOOKUP;
        }
        
        // 2. Course Information
        if (q.contains("ders") || q.contains("kredi") || q.contains("ects") || 
            q.contains("akts") || q.contains("önkoşul") || q.contains("dönem")) {
            return Intent.COURSE_INFO;
        }
        
        // 3. Policy / Exams / Internship / Graduation
        if (q.contains("yönetmelik") || q.contains("yönerge") || q.contains("sınav") || 
            q.contains("staj") || q.contains("mezuniyet") || q.contains("çap") || q.contains("yatay")) {
            return Intent.POLICY_FAQ;
        }
        
        // 4. Registration Procedures
        if (q.contains("kayıt") || q.contains("dondurma") || q.contains("harç")) {
            return Intent.REGISTRATION;
        }

        // Default: Unknown Intent
        return Intent.UNKNOWN;
    }
}