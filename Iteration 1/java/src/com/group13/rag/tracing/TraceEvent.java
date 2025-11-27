package com.group13.rag.tracing;

/**
 * Represents a single trace event within the RAG pipeline.
 * This Data Transfer Object (DTO) holds all metadata required for logging,
 * adhering to the specific format requested: {stage, inputs, outputsSummary, timingMs, errors}.
 */
public class TraceEvent {
    public String stage;           // The name of the pipeline stage (e.g., INTENT, RETRIEVE)
    public String inputs;          // Input data provided to this stage
    public String outputsSummary;  // Summary of the output produced by this stage
    public long timingMs;          // Execution time in milliseconds
    public String errors;          // Error message if failure occurred (optional, can be null)
    public long timestamp;         // System timestamp when the event was created

    /**
     * Full Constructor to initialize all fields.
     * Automatically sets the timestamp and handles null safety for strings.
     * @param stage The pipeline stage name.
     * @param inputs The input string (safe against null).
     * @param outputsSummary The output summary (safe against null).
     * @param timingMs Duration of the operation.
     * @param errors Error string (nullable).
     */
    public TraceEvent(String stage, String inputs, String outputsSummary, long timingMs, String errors) {
        this.stage = stage;
        // Ensure strings are not null to avoid NullPointerException during JSON processing
        this.inputs = inputs != null ? inputs : "";
        this.outputsSummary = outputsSummary != null ? outputsSummary : "";
        this.timingMs = timingMs;
        this.errors = errors;
        this.timestamp = System.currentTimeMillis();
    }
}