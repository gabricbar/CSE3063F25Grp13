import time
from .tracing import TraceBus
from .models import Answer  # for type hints (optional)

class RagOrchestrator:
    def __init__(self, intent_detector, query_writer, retriever, reranker, answer_agent, global_index):
        self.intent_detector = intent_detector
        self.query_writer = query_writer
        self.retriever = retriever
        self.reranker = reranker
        self.answer_agent = answer_agent
        self.global_index = global_index

    def run(self, user_question: str) -> Answer:
        """Run the pipeline for a single question and return an Answer object."""

        # START
        t0 = time.time()
        ...

        # INTENT
        t1 = time.time()
        intent = self.intent_detector.detect(user_question)
        TraceBus.push_full("INTENT", user_question, str(intent), int((time.time() - t1) * 1000))

        # QUERY
        t2 = time.time()
        terms = self.query_writer.write(user_question, intent)
        TraceBus.push_full("QUERY", user_question, str(terms), int((time.time() - t2) * 1000))

        # RETRIEVE
        t3 = time.time()
        hits = self.retriever.retrieve(terms, self.global_index)
        TraceBus.push_full("RETRIEVE", str(terms), f"{len(hits)} hits", int((time.time() - t3) * 1000))

        # RERANK
        t4 = time.time()
        reranked = self.reranker.rerank(terms, hits)
        best = reranked[0].score if reranked else 0
        TraceBus.push_full("RERANK", str(terms), f"best={best}", int((time.time() - t4) * 1000))

        # ANSWER
        t5 = time.time()
        answer = self.answer_agent.answer(user_question, reranked)
        TraceBus.push_full("ANSWER", user_question, answer.finalText[:80], int((time.time() - t5) * 1000))

        # END
        total = int((time.time() - t0) * 1000)
        TraceBus.push_full("END", "Pipeline completed", f"Total={total}ms", total)

        return answer

    def run_with_debug(self, user_question: str):
        """Run and also return intermediate artifacts useful for batch evaluation."""
        answer = self.run(user_question)
        # We can re-compute light debug info from traces? For now, return answer only.
        return answer
