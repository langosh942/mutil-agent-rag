from __future__ import annotations

import os
from typing import Dict, List, Optional

import streamlit as st

from multiagent_demo import DatabaseConfig, ModelConfig, MultiAgentOrchestrator, WeaviateConfig
from multiagent_demo.chunking import build_chunk_payloads, chunk_text
from multiagent_demo.database import DatabaseManager
from multiagent_demo.weaviate_client import WeaviateManager

st.set_page_config(page_title="多 Agent 编排 Demo", layout="wide")


# -----------------------------------------------------------------------------
# Session helpers
# -----------------------------------------------------------------------------

def _get_database_manager(config: DatabaseConfig) -> DatabaseManager:
    session_key = "db_manager"
    cfg_key = "db_config"

    stored_config: Optional[DatabaseConfig] = st.session_state.get(cfg_key)
    manager: Optional[DatabaseManager] = st.session_state.get(session_key)

    if manager is not None and stored_config == config:
        return manager

    warning_key = "db_warning"
    try:
        manager = DatabaseManager(config)
        st.session_state[session_key] = manager
        st.session_state[cfg_key] = config
        st.session_state.pop(warning_key, None)
        return manager
    except Exception as exc:  # pragma: no cover - depends on external DB
        fallback = DatabaseConfig()
        manager = DatabaseManager(fallback)
        st.session_state[session_key] = manager
        st.session_state[cfg_key] = fallback
        warning = f"数据库连接失败，已回退到 {fallback.url}: {exc}"
        st.session_state[warning_key] = warning
        return manager


def _get_weaviate_manager(config: WeaviateConfig) -> WeaviateManager:
    session_key = "weaviate_manager"
    cfg_key = "weaviate_config"

    stored_config: Optional[WeaviateConfig] = st.session_state.get(cfg_key)
    manager: Optional[WeaviateManager] = st.session_state.get(session_key)

    if manager is not None and stored_config == config:
        return manager

    manager = WeaviateManager(config)
    # Best-effort schema creation.
    manager.ensure_schema()

    st.session_state[session_key] = manager
    st.session_state[cfg_key] = config
    return manager


def _read_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.read()
    if not data:
        return ""
    for encoding in ("utf-8", "utf-16", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _model_config_from_inputs() -> ModelConfig:
    return ModelConfig(
        api_key=st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="支持 OpenAI 兼容接口，例如本地 vLLM / Azure OpenAI。",
        ),
        base_url=st.sidebar.text_input(
            "OpenAI Base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
        model=st.sidebar.text_input(
            "模型名称",
            value=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        ),
        temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.05),
        max_tokens=int(
            st.sidebar.number_input("Max Tokens", min_value=64, max_value=4096, value=512, step=32)
        ),
    )


def _weaviate_config_from_inputs() -> WeaviateConfig:
    return WeaviateConfig(
        url=st.sidebar.text_input("Weaviate URL", value=os.getenv("WEAVIATE_URL", "http://localhost:8080")),
        api_key=st.sidebar.text_input("Weaviate API Key", value=os.getenv("WEAVIATE_API_KEY", ""), type="password"),
        class_name=st.sidebar.text_input("Weaviate Class 名称", value="DocumentChunk"),
    )


def _database_config_from_inputs() -> DatabaseConfig:
    return DatabaseConfig(
        url=st.sidebar.text_input(
            "数据库连接串",
            value=os.getenv("DATABASE_URL", "sqlite:///demo.db"),
            help="输入 PostgreSQL 连接串，例如 postgresql+psycopg2://user:pass@host:5432/dbname。",
        ),
        echo=st.sidebar.checkbox("SQL Echo", value=False),
    )


def _show_sidebar_status(weaviate_manager: WeaviateManager) -> None:
    st.sidebar.markdown("---")
    if weaviate_manager.is_connected:
        st.sidebar.success("Weaviate 已连接，向量检索可用。")
    else:
        message = weaviate_manager.init_error or "未提供 Weaviate URL，将使用内存检索。"
        st.sidebar.warning(f"Weaviate 未连接：{message}")

    db_warning = st.session_state.get("db_warning")
    if db_warning:
        st.sidebar.error(db_warning)

    st.sidebar.markdown("---")
    if "chat_history" in st.session_state:
        st.sidebar.caption(f"当前会话轮数：{len(st.session_state['chat_history']) // 2}")


# -----------------------------------------------------------------------------
# Page layout
# -----------------------------------------------------------------------------

st.title("🧠 多 Agent 编排 Demo (LangGraph + Weaviate + PG)")
st.caption("上传文档 → 多路检索 → 工具/Web 搜索 → LLM 答复")

model_config = _model_config_from_inputs()
weaviate_config = _weaviate_config_from_inputs()
database_config = _database_config_from_inputs()

weaviate_manager = _get_weaviate_manager(weaviate_config)
db_manager = _get_database_manager(database_config)
_show_sidebar_status(weaviate_manager)

st.subheader("步骤 1：上传文档并写入知识库")
uploaded_files = st.file_uploader(
    "上传文本文件（支持 txt / md / csv / json）",
    type=["txt", "md", "csv", "json"],
    accept_multiple_files=True,
    help="文档会被自动切分并写入 Weaviate 与 数据库。",
)

ingest_col1, ingest_col2 = st.columns(2)
with ingest_col1:
    chunk_size = int(st.number_input("Chunk 大小", min_value=200, max_value=1200, value=500, step=50))
with ingest_col2:
    overlap = int(st.number_input("Chunk 重叠", min_value=0, max_value=300, value=120, step=10))

if st.button("写入知识库", use_container_width=True):
    if not uploaded_files:
        st.warning("请先选择至少一个文件。")
    else:
        ingested: List[Dict[str, int]] = []
        errors: List[str] = []
        for file in uploaded_files:
            text = _read_uploaded_file(file)
            if not text.strip():
                errors.append(f"文件 {file.name} 内容为空或无法读取。")
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            payloads = build_chunk_payloads(chunks, source=file.name)

            try:
                document_id = db_manager.ingest_document(file.name, payloads)
            except Exception as exc:  # pragma: no cover - depends on DB
                errors.append(f"写入数据库失败（{file.name}）：{exc}")
                continue

            for payload in payloads:
                payload["document_id"] = document_id

            weaviate_manager.upsert_texts(payloads)
            ingested.append({"name": file.name, "chunks": len(payloads)})

        if ingested:
            st.success(f"成功写入 {len(ingested)} 个文件。")
            st.json(ingested)
        if errors:
            st.error("\n".join(errors))

st.markdown("---")

st.subheader("步骤 2：提问并触发多 Agent 编排")
question = st.text_area("请输入问题", height=120)
run_button = st.button("运行多 Agent Workflow", use_container_width=True)

if run_button:
    if not question.strip():
        st.warning("请输入问题后再运行。")
    else:
        orchestrator = MultiAgentOrchestrator(
            model_config=model_config,
            weaviate_manager=weaviate_manager,
            database_manager=db_manager,
        )
        history = st.session_state.get("chat_history", [])
        result = orchestrator.run(question, chat_history=history)

        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": result.get("answer", "")})

        st.markdown("### 🤖 最终回答")
        st.write(result.get("answer", "未生成回答。"))

        col_vector, col_keyword = st.columns(2)
        with col_vector:
            with st.expander("向量检索结果", expanded=False):
                for item in result.get("vector_results", []):
                    st.markdown(
                        f"**source:** {item.get('source', 'N/A')} | **chunk:** {item.get('chunk_index')} | "
                        f"**score:** {item.get('score')}\n\n{item.get('text')}"
                    )
                if not result.get("vector_results"):
                    st.info("无向量检索结果。")
        with col_keyword:
            with st.expander("关键词检索结果", expanded=False):
                for item in result.get("keyword_results", []):
                    keywords = ", ".join(item.get("keywords", []) or [])
                    st.markdown(
                        f"**文档:** {item.get('document')} | **chunk:** {item.get('chunk_index')} | "
                        f"**关键词:** {keywords}\n\n{item.get('text')}"
                    )
                if not result.get("keyword_results"):
                    st.info("无关键词检索结果。")

        col_knowledge, col_tool = st.columns(2)
        with col_knowledge:
            with st.expander("知识图谱检索", expanded=False):
                for item in result.get("knowledge_results", []):
                    st.markdown(
                        f"**{item.get('subject')}** -{item.get('predicate')}→ **{item.get('object')}**"
                        f" （来源: {item.get('document')}）"
                    )
                if not result.get("knowledge_results"):
                    st.info("无知识图谱命中。")
        with col_tool:
            with st.expander("工具调用结果", expanded=False):
                for item in result.get("tool_outputs", []):
                    st.markdown(f"工具 {item.get('tool')} 输入 `{item.get('input')}` → 输出 `{item.get('output')}`")
                if not result.get("tool_outputs"):
                    st.info("未触发工具。")

        with st.expander("Web 搜索补充", expanded=False):
            for item in result.get("web_results", []):
                st.markdown(
                    f"[{item.get('title') or '无标题'}]({item.get('url') or '#'})\n\n"
                    f"{item.get('snippet') or '无摘要'}"
                )
            if not result.get("web_results"):
                st.info("无 Web 搜索命中。")

        if result.get("errors"):
            st.warning("; ".join(result["errors"]))

        st.markdown("---")

st.subheader("步骤 3：知识库概览")
with st.expander("已写入文档", expanded=False):
    try:
        docs = db_manager.list_documents()
        if docs:
            st.dataframe(docs)
        else:
            st.info("尚未有文档写入。")
    except Exception as exc:  # pragma: no cover - DB dependent
        st.error(f"无法读取文档列表：{exc}")

with st.expander("当前对话历史", expanded=False):
    history = st.session_state.get("chat_history", [])
    if not history:
        st.info("暂无对话记录。")
    else:
        for turn in history:
            st.markdown(f"**{turn.get('role')}**: {turn.get('content')}")
