package com.group13.rag.tracing;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A simple event bus implementing the Observer pattern for tracing.
 * It decouples the event producers (pipeline stages) from consumers (loggers).
 */

public class TraceBus {
    // List of registered listeners (observers)
    private static final List<Consumer<TraceEvent>> listeners = new ArrayList<>();

    /**
     * Registers a new listener to receive trace events.
     * @param listener The consumer to accept TraceEvents.
     */
    public static void register(Consumer<TraceEvent> listener) {
        listeners.add(listener);
    }

    /**
     * 1. BACKWARD COMPATIBILITY MODE
     * This method allows existing components to call 'push(stage, message)' without breaking changes.
     * It treats the 'message' as the 'outputsSummary' and defaults timing to 0.
     */
    public static void push(String stage, String message) {
        pushFull(stage, "", message, 0, null);
    }

    /**
     * 2. FULL DETAIL MODE
     * Pushes a complete trace event containing all fields required by the project specifications.
     *
     * @param stage The pipeline stage name (e.g., INTENT, RETRIEVE).
     * @param inputs The input data for this stage.
     * @param outputsSummary A summary of the output produced by this stage.
     * @param timingMs Execution time in milliseconds.
     * @param errors Error message, if any (can be null).
     */
    public static void pushFull(String stage, String inputs, String outputsSummary, long timingMs, String errors) {
        TraceEvent event = new TraceEvent(stage, inputs, outputsSummary, timingMs, errors);
        for (Consumer<TraceEvent> listener : listeners) {
            listener.accept(event);
        }
    }
}