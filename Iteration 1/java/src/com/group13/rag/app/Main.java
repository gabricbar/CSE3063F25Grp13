package com.group13.rag.app;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.group13.rag.core.*;
import com.group13.rag.impl.*;
import com.group13.rag.model.*;
import com.group13.rag.pipeline.RagOrchestrator;
import com.group13.rag.tracing.TraceBus;
import com.group13.rag.tracing.JsonlTraceSink;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        // --- 1. ARGUMENT CHECK ---
        String questionArg = null;

        for (int i = 0; i < args.length; i++) {
            if ("--q".equals(args[i]) && i + 1 < args.length) {
                questionArg = args[i + 1];
            }
        }

        // If no question is provided, print error and exit
        if (questionArg == null || questionArg.trim().isEmpty()) {
            System.err.println("ERROR: Question not specified! Usage: java -jar rag.jar --q \"Question...\"");
            return;
        }

        // --- 2. LOAD DATA ---
        List<Chunk> chunks = null;
        KeywordIndex index = null;
        Gson gson = new Gson();

        try {
            Type chunkListType = new TypeToken<List<Chunk>>(){}.getType();
            chunks = gson.fromJson(new FileReader("data/chunks.json"), chunkListType);
            index = gson.fromJson(new FileReader("data/index.json"), KeywordIndex.class);
        } catch (IOException e) {
            System.err.println("CRITICAL ERROR: 'data/chunks.json' or 'data/index.json' not found.");
            return;
        }

        // --- 3. LOGGING (Write to file, no console output) ---
        File logDir = new File("logs");
        if (!logDir.exists()) logDir.mkdir();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd-HHmmss");
        String timestamp = sdf.format(new Date());
        
        // SINK 1: Günlük, zaman damgalı log dosyası (logs klasörüne)
        String dailyLogFileName = "logs/run-" + timestamp + ".jsonl";
        TraceBus.register(new JsonlTraceSink(dailyLogFileName));
        
        // SINK 2: Genel, sabit isimli log dosyası (Proje ana dizinine)
        // Bu dosya her çalıştırmada üzerine ekler (append).
        String generalLogFileName = "rag_trace.jsonl"; // <-- YOL DÜZELTİLDİ
        TraceBus.register(new JsonlTraceSink(generalLogFileName));


        // --- 4. INSTANTIATE COMPONENTS ---
        IntentDetector intentDetector = new RuleBasedIntentDetector();
        QueryWriter queryWriter = new HeuristicQueryWriter();
        Retriever retriever = new KeywordRetriever(); 
        Reranker reranker = new SimpleReranker(chunks); 
        AnswerAgent answerAgent = new TemplateAnswerAgent(); 

        // --- 5. ORCHESTRATION ---
        RagOrchestrator orchestrator = new RagOrchestrator(
            intentDetector, queryWriter, retriever, reranker, answerAgent, index
        );

        // --- 6. RUN AND PRINT ANSWER ---
        // Orchestrator returns a String
        String response = orchestrator.run(questionArg);
        
        // stdout: Only the final answer
        System.out.println(response);
    }
}