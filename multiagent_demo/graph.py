from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .agents import run_tooling, web_search
from .database import DatabaseManager
from .llm import build_chat_model
from .weaviate_client import WeaviateManager


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    vector_results: List[Dict[str, Any]]
    keyword_results: List[Dict[str, Any]]
    knowledge_results: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    answer: str
    chat_history: List[Dict[str, str]]
    errors: List[str]


class MultiAgentOrchestrator:
    """Constructs and executes the LangGraph-based multi-agent workflow."""

    def __init__(
        self,
        model_config: ModelConfig,
        weaviate_manager: WeaviateManager,
        database_manager: DatabaseManager,
        *,
        top_k: int = 4,
    ) -> None:
        self.model_config = model_config
        self.weaviate_manager = weaviate_manager
        self.database_manager = database_manager
        self.top_k = top_k

        llm_resources = build_chat_model(model_config)
        self.chat_model = llm_resources.model
        self.llm_error = llm_resources.error
        self.rewrite_chain = None
        self.answer_chain = None
        if self.chat_model is not None:
            self._build_llm_chains()

        self.workflow = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> AgentState:
        initial_state: AgentState = {
            "question": question,
            "chat_history": chat_history or [],
            "vector_results": [],
            "keyword_results": [],
            "knowledge_results": [],
            "tool_outputs": [],
            "web_results": [],
            "errors": [self.llm_error] if self.llm_error else [],
        }

        return self.workflow.invoke(initial_state)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("rewrite", self._rewrite_question)
        graph.add_node("vector", self._vector_search)
        graph.add_node("keyword", self._keyword_search)
        graph.add_node("knowledge", self._knowledge_search)
        graph.add_node("tool", self._tool_node)
        graph.add_node("web", self._web_search_node)
        graph.add_node("answer", self._answer_node)

        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "vector")
        graph.add_edge("vector", "keyword")
        graph.add_edge("keyword", "knowledge")
        graph.add_edge("knowledge", "tool")
        graph.add_edge("tool", "web")
        graph.add_edge("web", "answer")
        graph.add_edge("answer", END)

        return graph.compile()

    def _build_llm_chains(self) -> None:
        history_placeholder = "{history}"
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You help transform user questions into search-friendly queries. "
                    "Keep critical details from the conversation history if present.",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\nOriginal question: {question}\n\nRewrite the question to maximise retrieval quality.",
                ),
            ]
        )
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the supplied retrieval context, knowledge graph entries, "
                    "tool outputs and web search results to answer the question. If context is lacking, "
                    "be explicit about uncertainties.",
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}\n\nKnowledge graph:\n{knowledge}\n\nTool outputs:\n{tools}\n\nWeb search:\n{web}\n\nCompose a concise answer in the same language as the question.",
                ),
            ]
        )
        parser = StrOutputParser()
        self.rewrite_chain = rewrite_prompt | self.chat_model | parser
        self.answer_chain = answer_prompt | self.chat_model | parser

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _rewrite_question(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        if not question:
            return {"rewritten_question": ""}

        history_text = self._history_to_text(state.get("chat_history", []))

        if self.rewrite_chain is None:
            return {"rewritten_question": question}

        try:
            rewritten = self.rewrite_chain.invoke({"question": question, "history": history_text})
            rewritten = rewritten.strip() or question
            return {"rewritten_question": rewritten}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Rewrite agent error: {exc}")
            return {"rewritten_question": question, "errors": errors}

    def _vector_search(self, state: AgentState) -> AgentState:
        query_text = state.get("rewritten_question") or state.get("question") or ""
        try:
            results = self.weaviate_manager.query(query_text, limit=self.top_k)
            return {"vector_results": results}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Vector search error: {exc}")
            return {"vector_results": [], "errors": errors}

    def _keyword_search(self, state: AgentState) -> AgentState:
        query_text = state.get("rewritten_question") or state.get("question") or ""
        try:
            results = self.database_manager.keyword_search(query_text, limit=self.top_k)
            return {"keyword_results": results}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Keyword search error: {exc}")
            return {"keyword_results": [], "errors": errors}

    def _knowledge_search(self, state: AgentState) -> AgentState:
        query_text = state.get("rewritten_question") or state.get("question") or ""
        try:
            results = self.database_manager.knowledge_graph_search(query_text, limit=self.top_k)
            return {"knowledge_results": results}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Knowledge graph search error: {exc}")
            return {"knowledge_results": [], "errors": errors}

    def _tool_node(self, state: AgentState) -> AgentState:
        text = state.get("rewritten_question") or state.get("question") or ""
        outputs, tool_error = run_tooling(text)
        errors = list(state.get("errors", []))
        if tool_error:
            errors.append(tool_error)
        if outputs:
            return {"tool_outputs": outputs, "errors": errors} if errors else {"tool_outputs": outputs}
        if errors:
            return {"tool_outputs": [], "errors": errors}
        return {"tool_outputs": []}

    def _web_search_node(self, state: AgentState) -> AgentState:
        text = state.get("rewritten_question") or state.get("question") or ""
        results, web_error = web_search(text, max_results=self.top_k)
        errors = list(state.get("errors", []))
        if web_error:
            errors.append(f"Web search error: {web_error}")
        if errors:
            return {"web_results": results, "errors": errors}
        return {"web_results": results}

    def _answer_node(self, state: AgentState) -> AgentState:
        context = self._format_vector_results(state.get("vector_results", []))
        keyword_context = self._format_keyword_results(state.get("keyword_results", []))
        knowledge_context = self._format_knowledge_results(state.get("knowledge_results", []))
        tools_context = self._format_tool_outputs(state.get("tool_outputs", []))
        web_context = self._format_web_results(state.get("web_results", []))

        compiled_context = "\n".join(filter(None, [context, keyword_context])) or "无内部知识库命中。"
        compiled_knowledge = knowledge_context or "无知识图谱命中。"
        compiled_tools = tools_context or "无工具调用结果。"
        compiled_web = web_context or "无 Web 搜索命中。"

        question = state.get("question", "")
        if self.answer_chain is None:
            answer = self._fallback_answer(question, compiled_context, compiled_knowledge, compiled_tools, compiled_web)
            return {"answer": answer}

        try:
            answer = self.answer_chain.invoke(
                {
                    "question": question,
                    "context": compiled_context,
                    "knowledge": compiled_knowledge,
                    "tools": compiled_tools,
                    "web": compiled_web,
                }
            )
            return {"answer": answer}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Answer generation error: {exc}")
            fallback = self._fallback_answer(question, compiled_context, compiled_knowledge, compiled_tools, compiled_web)
            return {"answer": fallback, "errors": errors}

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _history_to_text(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(no previous turns)"
        return "\n".join(f"{item.get('role')}: {item.get('content')}" for item in history)

    @staticmethod
    def _format_vector_results(items: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(items, start=1):
            text = item.get("text") or ""
            source = item.get("source") or ""
            score = item.get("score")
            prefix = f"[向量#{idx}]"
            if source:
                prefix += f" ({source})"
            if score is not None:
                prefix += f" score={score}"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_keyword_results(items: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(items, start=1):
            doc = item.get("document") or ""
            text = item.get("text") or ""
            keywords = ", ".join(item.get("keywords", []) or [])
            lines.append(f"[关键词#{idx}] {doc} -> {text} | keywords: {keywords}")
        return "\n".join(lines)

    @staticmethod
    def _format_knowledge_results(items: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(items, start=1):
            subject = item.get("subject")
            predicate = item.get("predicate")
            obj = item.get("object")
            doc = item.get("document") or ""
            lines.append(f"[知识#{idx}] {subject} -{predicate}-> {obj} (source: {doc})")
        return "\n".join(lines)

    @staticmethod
    def _format_tool_outputs(items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = []
        for item in items:
            tool = item.get("tool")
            tool_input = item.get("input")
            tool_output = item.get("output")
            lines.append(f"{tool}: {tool_input} => {tool_output}")
        return "\n".join(lines)

    @staticmethod
    def _format_web_results(items: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(items, start=1):
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            url = item.get("url") or ""
            lines.append(f"[Web#{idx}] {title} | {snippet} | {url}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(
        question: str,
        context: str,
        knowledge: str,
        tools: str,
        web: str,
    ) -> str:
        return (
            "（Fallback 模式）\n"
            f"问题：{question}\n\n"
            "根据检索到的内容整理如下：\n"
            f"上下文：\n{context}\n\n"
            f"知识图谱：\n{knowledge}\n\n"
            f"工具调用：\n{tools}\n\n"
            f"Web 搜索：\n{web}\n"
        )
