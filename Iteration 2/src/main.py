import argparse
import json
import os
import time
import sys
from datetime import datetime

# Pipeline, Factory and Tracing components
from src.pipeline import RagOrchestrator
from src.factory import PipelineFactory
from src.tracing import TraceBus, JsonlTraceSink

def setup_tracing():
    """Initializes the tracing system and registers the JSONL sink."""
    try:
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = f"logs/run-{timestamp}.jsonl"
        TraceBus.register(JsonlTraceSink(log_path))
    except Exception as e:
        print(f"Critical Error: Could not initialize tracing. {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    
    # Configuration is required
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    
    # Reranker override (Optional - overrides config file setting)
    parser.add_argument("--reranker", choices=["simple", "cosine", "hybrid"], help="Override reranker type")

    # Mode Selection: Single question or Batch processing
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--q", help="Single query string")
    group.add_argument("--batch", help="Path to input JSONL file for batch processing")

    # Output file (Optional for batch mode)
    parser.add_argument("--out", help="Path to output JSONL file for results")

    args = parser.parse_args()

    # --- 1. CONFIGURATION LOADING ---
    config = {}
    try:
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' not found.")
            return

        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Config file is not a valid JSON. {e}")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # --- 2. CLI STRATEGY OVERRIDE ---
    # Implements OCP by allowing strategy swapping without code changes
    if args.reranker:
        print(f"⚠️ [CLI Override] Reranker type changed to: {args.reranker.upper()}")
        config.setdefault("pipeline", {}).setdefault("reranker", {})["type"] = args.reranker

    # --- 3. INITIALIZE TRACING ---
    setup_tracing()

    # --- 4. PIPELINE CONSTRUCTION ---
    try:
        if not args.out or args.q:
            print(f"Initializing RagOrchestrator with config: {args.config}...")
        
        pipeline = PipelineFactory.create(config)
        
        if not args.out or args.q:
            print("✅ Pipeline ready.\n")
    except Exception as e:
        print(f"❌ Error: Could not construct the pipeline. Reason: {e}")
        return

    # --- MODE A: SINGLE QUESTION (--q) ---
    if args.q:
        try:
            print("=" * 60)
            print(f"QUERY: {args.q}")
            print("-" * 60)
            
            start_t = time.time()
            # Execution using the Orchestrator (Controller)
            answer = pipeline.run(args.q) 
            end_t = time.time()
            
            # Extract final text using fallback attributes
            final_text = getattr(answer, 'finalText', getattr(answer, 'text', str(answer)))

            print(f"ANSWER: {final_text}")
            print("\nCITATIONS:")
            if hasattr(answer, 'citations') and answer.citations:
                for cit in answer.citations:
                    # Formatted according to handout: docid:section:span
                    print(f" - {str(cit)}")
            else:
                print(" - No citations available.")
                
            print(f"\nLatency: {(end_t - start_t)*1000:.2f} ms")
            print("=" * 60)
        except Exception as e:
            print(f"Error during execution: {e}")

    # --- MODE B: BATCH PROCESSING (--batch) ---
    elif args.batch:
        fout = None
        try:
            if not os.path.exists(args.batch):
                print(f"Error: Batch file not found -> {args.batch}")
                return

            if args.out:
                print(f"Batch processing started: {args.batch} -> {args.out}")
                fout = open(args.out, "w", encoding="utf-8")
            else:
                print(f"Warning: No --out parameter provided. Results will be printed to stdout.\n")

            with open(args.batch, "r", encoding="utf-8") as fin:
                lines = fin.readlines()
                total = len(lines)
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Line {i+1}: Invalid JSON format, skipping.")
                        continue
                    
                    # Extract query from common keys
                    q_text = data.get("question") or data.get("text") or data.get("q")
                    q_id = data.get("id", str(i+1))
                    
                    if not q_text: continue

                    # Execute query and measure latency
                    start_t = time.time()
                    result = pipeline.run(q_text)
                    end_t = time.time()

                    # Data Extraction and Formatting
                    ans_text = getattr(result, 'finalText', getattr(result, 'text', ""))
                    citations = [str(c) for c in (getattr(result, 'citations', []) or [])]

                    output_record = {
                        "id": q_id,
                        "question": q_text,
                        "answer": ans_text,
                        "citations": citations,
                        "latency_ms": int((end_t - start_t) * 1000)
                    }
                    
                    # Output Strategy
                    if fout:
                        fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                        if (i+1) % 5 == 0:
                            print(f"Progress: {i+1}/{total}")
                    else:
                        print("-" * 50)
                        print(f"QUERY [{q_id}]: {q_text}")
                        print(f"ANSWER: {ans_text}")
                        print(f"SOURCES: {citations}")
                        print("-" * 50)

            print("✅ Batch processing completed successfully.")

        except Exception as batch_err:
            print(f"Critical Batch Error: {batch_err}")
        finally:
            if fout:
                fout.close()

if __name__ == "__main__":
    main()