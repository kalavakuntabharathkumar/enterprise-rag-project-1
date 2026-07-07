"""LangGraph multi-agent state graph for the RAG pipeline.

Topology
--------
                       ┌──────────────┐
                       │ router_agent │
                       └──────┬───────┘
              ┌───────────────┼───────────────┐
           (rag)           (tool)        (small_talk)
              │               │               │
    ┌─────────▼──────┐ ┌──────▼──────┐        │
    │ retrieval_agent│ │  tool_agent │        │
    └─────────┬──────┘ └──────┬──────┘        │
              └───────┬───────┘               │
                      ▼                       ▼
                ┌─────────────┐         ┌─────────────┐
                │ answer_agent│◄────────┤ answer_agent│
                └──────┬──────┘         └─────────────┘
                       ▼
                      END

Memory
------
MemorySaver checkpoints the full state after every node, keyed by
thread_id.  Passing the same thread_id on consecutive /ask requests
gives the graph access to the prior conversation turns.
"""

import asyncio
import re
from typing import Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.config import Config
from backend.inference_client import InferenceError, generate as llm_generate
from backend.logger import app_logger
from backend.tools import dispatch_tool
from ml.intent_classifier import IntentClassifier
from ml.reranker import LexicalReranker
from models.intent_classifier import QueryTypeClassifier


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    intent: str                  # "rag" | "tool" | "small_talk"
    tool_name: Optional[str]
    tool_input: Optional[str]
    tool_result: Optional[str]
    retrieved_docs: List[Dict]   # [{"text": str, "source": str, "score": float}]
    context: str
    answer: str
    sources: List[str]
    confidence: float
    query_type: Optional[str]
    session_id: Optional[str]
    top_similarity: float
    avg_similarity: float
    llm_skipped: bool
    estimated_tokens: int


# ---------------------------------------------------------------------------
# Shared components — constructed once at module load, reused across calls
# ---------------------------------------------------------------------------

_intent_classifier = IntentClassifier()
_query_type_classifier = QueryTypeClassifier()
_reranker = LexicalReranker()

# The DocumentRetriever is injected by RAGPipeline after it is created so
# graph.py never imports RAGPipeline (which would create a circular dep).
_retriever = None


def set_retriever(retriever) -> None:
    """Inject the live DocumentRetriever instance into the graph."""
    global _retriever
    _retriever = retriever


CHARS_PER_TOKEN = 4

# Regex that detects calculator-style queries: a digit followed by arithmetic
# operators/parentheses, optionally preceded by trigger words.
_CALC_RE = re.compile(
    r"(?:calculate|compute|what(?:'?s| is)\s+)?\s*"
    r"\d[\d\s\.\+\-\*\/\(\)\^\%]*[\d\)]",
    re.IGNORECASE,
)

_CANNED: Dict[str, str] = {
    "greeting": "Hello! Upload a PDF and ask me a question about it whenever you're ready.",
    "chit_chat": "I'm a document assistant focused on the PDFs you upload — ask me something about their content.",
}


# ---------------------------------------------------------------------------
# Node: router_agent
# ---------------------------------------------------------------------------

def router_agent(state: AgentState) -> AgentState:
    """Classify the incoming question and decide which agent handles it next.

    Decision order
    --------------
    1. Non-document intent (greeting / chit_chat) → small_talk
    2. Arithmetic expression → tool (calculator)
    3. Everything else → rag
    """
    question = state["question"]
    intent_label = _intent_classifier.predict(question)

    if intent_label != "document_question":
        return {**state, "intent": "small_talk", "query_type": None}

    if _CALC_RE.search(question):
        match = re.search(r"[\d][\d\s\.\+\-\*\/\(\)\^\%]*", question)
        expr = match.group(0).strip() if match else question
        return {
            **state,
            "intent": "tool",
            "tool_name": "calculator",
            "tool_input": expr,
            "query_type": None,
        }

    query_type = _query_type_classifier.predict(question)
    return {**state, "intent": "rag", "query_type": query_type}


# ---------------------------------------------------------------------------
# Node: retrieval_agent
# ---------------------------------------------------------------------------

async def retrieval_agent(state: AgentState) -> AgentState:
    """Run FAISS retrieval and lexical re-ranking, populate retrieved_docs."""
    if _retriever is None:
        app_logger.error("retrieval_agent: DocumentRetriever has not been injected")
        return {
            **state,
            "retrieved_docs": [],
            "context": "",
            "sources": [],
            "top_similarity": 0.0,
            "avg_similarity": 0.0,
        }

    question = state["question"]
    k = Config.TOP_K

    # Over-fetch candidates so the re-ranker has material to work with.
    candidates = await asyncio.to_thread(
        _retriever.retrieve_with_scores, question, k * 3
    )

    if not candidates:
        return {
            **state,
            "retrieved_docs": [],
            "context": "",
            "sources": [],
            "top_similarity": 0.0,
            "avg_similarity": 0.0,
        }

    doc_texts = [doc.page_content for doc, _ in candidates]
    embedding_scores = [score for _, score in candidates]
    order = _reranker.rerank(
        question, doc_texts, embedding_scores, alpha=Config.RERANK_ALPHA
    )

    top_indices = order[:k]
    selected = [candidates[i] for i in top_indices]
    docs = [doc for doc, _ in selected]
    similarities = [score for _, score in selected]

    retrieved = [
        {
            "text": docs[i].page_content,
            "source": docs[i].metadata.get("source", "unknown"),
            "score": float(similarities[i]),
        }
        for i in range(len(docs))
    ]

    context = "\n\n".join(r["text"] for r in retrieved)
    sources = list(dict.fromkeys(r["source"] for r in retrieved))
    top_sim = max(similarities) if similarities else 0.0
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    return {
        **state,
        "retrieved_docs": retrieved,
        "context": context,
        "sources": sources,
        "top_similarity": top_sim,
        "avg_similarity": avg_sim,
    }


# ---------------------------------------------------------------------------
# Node: tool_agent
# ---------------------------------------------------------------------------

def tool_agent(state: AgentState) -> AgentState:
    """Execute the requested tool and store the result in tool_result."""
    tool_name = state.get("tool_name") or ""
    tool_input = state.get("tool_input") or state["question"]
    result = dispatch_tool(tool_name, tool_input)
    return {**state, "tool_result": result}


# ---------------------------------------------------------------------------
# Node: answer_agent
# ---------------------------------------------------------------------------

async def answer_agent(state: AgentState) -> AgentState:
    """Produce the final answer from context (RAG), tool output, or canned reply.

    For RAG queries the LLM call is run in a thread pool so it does not
    block the event loop.
    """
    intent = state["intent"]
    question = state["question"]

    # ── Small-talk ──────────────────────────────────────────────────────────
    if intent == "small_talk":
        label = _intent_classifier.predict(question)
        canned = _CANNED.get(label, _CANNED["chit_chat"])
        return {
            **state,
            "answer": canned,
            "confidence": 1.0,
            "sources": [],
            "llm_skipped": True,
            "estimated_tokens": 0,
        }

    # ── Tool ────────────────────────────────────────────────────────────────
    if intent == "tool":
        tool_result = state.get("tool_result") or ""
        if tool_result.startswith("Error"):
            answer = tool_result
        else:
            answer = f"The result is {tool_result}."
        return {
            **state,
            "answer": answer,
            "confidence": 1.0,
            "sources": [],
            "llm_skipped": True,
            "estimated_tokens": 0,
        }

    # ── RAG ─────────────────────────────────────────────────────────────────
    context = state.get("context", "")
    if not context:
        return {
            **state,
            "answer": "No relevant information found. Please upload a PDF first.",
            "confidence": 0.0,
            "sources": [],
            "llm_skipped": True,
            "estimated_tokens": 0,
        }

    retrieved_docs = state.get("retrieved_docs", [])
    top_similarity = state.get("top_similarity", 0.0)
    query_type = state.get("query_type", "factual")

    # Direct-return shortcut: high-confidence factual hit skips the LLM.
    if query_type == "factual" and top_similarity >= Config.FACTUAL_DIRECT_THRESHOLD:
        answer = retrieved_docs[0]["text"].strip()
        sources = list(dict.fromkeys(r["source"] for r in retrieved_docs))
        confidence = _estimate_confidence(answer, context)
        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "llm_skipped": True,
            "estimated_tokens": 0,
        }

    prompt = _build_prompt(question, context, query_type)

    try:
        answer = await asyncio.to_thread(llm_generate, prompt)
    except InferenceError:
        raise

    confidence = _estimate_confidence(answer, context)

    if confidence < Config.CONFIDENCE_THRESHOLD:
        answer = (
            "I'm not confident in this answer based on the available information. "
            "Please rephrase or provide more context."
        )

    sources = list(dict.fromkeys(r["source"] for r in retrieved_docs))
    estimated_tokens = (len(prompt) + len(answer)) // CHARS_PER_TOKEN

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "llm_skipped": False,
        "estimated_tokens": estimated_tokens,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context: str, query_type: str) -> str:
    if query_type == "summarization":
        instruction = (
            "Provide a comprehensive summary based solely on the context below."
        )
    elif query_type == "comparison":
        instruction = (
            "Compare and contrast the relevant aspects described in the context below."
        )
    else:
        instruction = (
            "Answer the question based solely on the context below. "
            "If the answer is not present in the context, say so."
        )

    return (
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def _estimate_confidence(answer: str, context: str) -> float:
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words & context_words)
    total = len(answer_words)
    return min(overlap / total, 1.0) if total else 0.0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_from_router(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "rag":
        return "retrieval_agent"
    if intent == "tool":
        return "tool_agent"
    return "answer_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the LangGraph StateGraph with MemorySaver."""
    builder = StateGraph(AgentState)

    builder.add_node("router_agent", router_agent)
    builder.add_node("retrieval_agent", retrieval_agent)
    builder.add_node("tool_agent", tool_agent)
    builder.add_node("answer_agent", answer_agent)

    builder.set_entry_point("router_agent")

    builder.add_conditional_edges(
        "router_agent",
        _route_from_router,
        {
            "retrieval_agent": "retrieval_agent",
            "tool_agent": "tool_agent",
            "answer_agent": "answer_agent",
        },
    )

    builder.add_edge("retrieval_agent", "answer_agent")
    builder.add_edge("tool_agent", "answer_agent")
    builder.add_edge("answer_agent", END)

    return builder.compile(checkpointer=MemorySaver())


# Module-level compiled graph — imported by RAGPipeline.
graph = build_graph()
