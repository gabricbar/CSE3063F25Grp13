package com.group13.rag.tracing;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.function.Consumer;

/**
 * A trace sink that writes events to a file in JSONL (JSON Lines) format.
 * Each event is written as a single-line JSON object.
 */
public class JsonlTraceSink implements Consumer<TraceEvent> {
    private String logFilePath;

    public JsonlTraceSink(String logFilePath) {
        this.logFilePath = logFilePath;
    }

    @Override
    public void accept(TraceEvent event) {
        // Manual JSON construction to adhere to the required format:
        // {stage, inputs, outputsSummary, timingMs, errors?}
        
        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"stage\": \"").append(event.stage).append("\", ");
        
        // Sanitize strings: Replace double quotes with single quotes to prevent JSON syntax errors.
        // Also replace newlines with spaces to ensure the log remains on a single line.
        String safeInputs = event.inputs.replace("\"", "'").replace("\n", " ");
        String safeOutputs = event.outputsSummary.replace("\"", "'").replace("\n", " ");
        
        json.append("\"inputs\": \"").append(safeInputs).append("\", ");
        json.append("\"outputsSummary\": \"").append(safeOutputs).append("\", ");
        json.append("\"timingMs\": ").append(event.timingMs);

        // Append the 'errors' field only if it is present and not empty
        if (event.errors != null && !event.errors.isEmpty()) {
            json.append(", \"errors\": \"").append(event.errors.replace("\"", "'")).append("\"");
        }

        json.append("}");

        // Write the JSON string to the log file in append mode
        try (PrintWriter out = new PrintWriter(new FileWriter(logFilePath, true))) {
            out.println(json.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}