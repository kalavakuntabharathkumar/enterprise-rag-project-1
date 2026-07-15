import time
from typing import Optional

from langchain_core.documents import Document

from analytics.query_log import QueryAnalytics
from backend.config import Config
from backend.graph import AgentState, graph, set_retriever
from backend.inference_client import InferenceError
from backend.latency import record_generation_time, record_retrieval_time
from backend.logger import app_logger, query_logger
from backend.retriever import DocumentRetriever
from backend.utils import extract_text_from_pdf
from preprocessing.pipeline import PreprocessingPipeline


class RAGPipeline:
    def __init__(self):
        # Retriever uses lazy embeddings — safe to construct at import time
        # even when OPENAI_API_KEY is not yet set.
        self.retriever = DocumentRetriever()
        self.preprocessor = PreprocessingPipeline(min_quality=Config.MIN_CHUNK_QUALITY)
        self.analytics = QueryAnalytics()

        # Inject this retriever into the shared graph so retrieval_agent
        # can call FAISS without importing RAGPipeline.
        set_retriever(self.retriever)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def process_pdf(self, file_path: str):
        """Process PDF: extract, clean, chunk, dedupe, score, embed, store."""
        try:
            text = extract_text_from_pdf(file_path)
            records = self.preprocessor.run(text, source=file_path)

            documents = [
                Document(page_content=r["text"], metadata=r["metadata"])
                for r in records
            ]

            if not documents:
                app_logger.warning(
                    f"No indexable chunks survived preprocessing for {file_path}"
                )
                return {
                    "status": "error",
                    "message": (
                        "No usable text could be extracted from this PDF after "
                        "cleaning and quality filtering."
                    ),
                    "chunks_indexed": 0,
                }

            self.retriever.add_documents(documents)

            app_logger.info(
                f"Processed PDF: {file_path} ({len(documents)} chunks indexed)"
            )
            return {
                "status": "success",
                "message": "PDF processed successfully",
                "chunks_indexed": len(documents),
            }
        except Exception as e:
            app_logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "message": str(e), "chunks_indexed": 0}

    # ------------------------------------------------------------------
    # Query answering — delegates to the LangGraph multi-agent pipeline
    # ------------------------------------------------------------------

    async def ask_question(
        self, question: str, session_id: Optional[str] = None
    ) -> dict:
        """Route and answer *question* through the LangGraph state graph.

        The graph handles intent routing, retrieval, optional tool calls,
        and generation.  MemorySaver checkpoints the full state so that
        repeated calls with the same *session_id* retain conversation
        history across requests.
        """
        start = time.time()

        thread_id = session_id or "default"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: AgentState = {
            "question": question,
            "intent": "",
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "query_type": None,
            "session_id": session_id,
            "top_similarity": 0.0,
            "avg_similarity": 0.0,
            "llm_skipped": True,
            "estimated_tokens": 0,
        }

        try:
            result: AgentState = await graph.ainvoke(initial_state, config=config)
        except InferenceError:
            raise
        except Exception as e:
            app_logger.error(f"Graph invocation error: {e}")
            latency_ms = (time.time() - start) * 1000
            self._log_query(
                question, "error", None, [], 0.0, 0.0, 0.0, latency_ms,
                answered=False, llm_skipped=True, estimated_tokens=0,
                estimated_cost_usd=0.0, llm_backend="error",
            )
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "intent": "error",
                "query_type": None,
            }

        latency_ms = (time.time() - start) * 1000

        intent = result.get("intent", "")
        query_type = result.get("query_type")
        retrieved_docs = result.get("retrieved_docs", [])
        top_similarity = result.get("top_similarity", 0.0)
        avg_similarity = result.get("avg_similarity", 0.0)
        confidence = result.get("confidence", 0.0)
        llm_skipped = result.get("llm_skipped", True)
        estimated_tokens = result.get("estimated_tokens", 0)
        answer = result.get("answer", "")

        # Annotate latency breakdown for the middleware recorder.
        # Retrieval and generation phases are now internal to the graph;
        # we record the full round-trip so /analytics/summary remains useful.
        record_retrieval_time(latency_ms * 0.5 if not llm_skipped else latency_ms)
        record_generation_time(latency_ms * 0.5 if not llm_skipped else 0.0)

        query_logger.info(
            f"Question: {question} | Intent: {intent} | "
            f"Confidence: {confidence:.3f} | Latency: {latency_ms:.0f}ms"
        )

        self._log_query(
            question, intent, query_type,
            retrieved_docs, top_similarity, avg_similarity,
            confidence, latency_ms,
            answered=bool(answer),
            llm_skipped=llm_skipped,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=0.0,
            llm_backend="none" if llm_skipped else Config.LLM_BACKEND,
        )

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "confidence": confidence,
            "intent": intent,
            "query_type": query_type,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_query(
        self, question, intent, query_type, docs, top_similarity, avg_similarity,
        confidence, latency_ms, answered, llm_skipped, estimated_tokens,
        estimated_cost_usd, llm_backend="unknown",
    ):
        try:
            num_docs = len(docs) if isinstance(docs, list) else 0
            self.analytics.log({
                "timestamp": time.time(),
                "question": question,
                "intent": intent,
                "query_type": query_type,
                "num_docs_retrieved": num_docs,
                "top_similarity": top_similarity,
                "avg_similarity": avg_similarity,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "answered": answered,
                "llm_skipped": llm_skipped,
                "estimated_tokens": estimated_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "llm_backend": llm_backend,
            })
        except Exception as e:
            app_logger.warning(f"Failed to log query analytics: {e}")
