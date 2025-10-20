# 多 Agent 编排 Demo（LangGraph + Weaviate + PostgreSQL）

本仓库实现了一个使用 **Python / LangGraph** 构建的多 Agent 编排示例。整套流程涵盖了用户问题输入、问题改写、向量检索、关键词检索、知识图谱检索、工具调用、Web 搜索以及最终的 LLM 问答。数据接入方面同时支持 **Weaviate 向量数据库** 与 **PostgreSQL**（默认可回落到本地 SQLite），前端界面由 Streamlit 提供，实现了文件上传、问答交互和模型配置（兼容 OpenAI API 格式）。

> 🇬🇧 English summary: This repository delivers a multi-agent orchestration demo powered by Python, LangGraph, Weaviate and PostgreSQL. It exposes a Streamlit UI with file upload, Q&A and OpenAI-compatible model configuration. The agent workflow covers query rewriting, vector/keyword/knowledge-graph retrieval, tool use, web search and LLM answering.

## 功能特性

- ✅ LangGraph 构建的多 Agent 工作流：
  - 用户输入处理
  - 问题改写（LLM）
  - 向量检索（Weaviate，可回退离线检索）
  - 关键词检索（PostgreSQL / SQLite）
  - 知识图谱检索（基于文档关键词构造的三元组）
  - 工具调用（示例内置计算器）
  - Web 搜索（DuckDuckGo）
  - LLM 问答总结
- ✅ Streamlit 前端：文件上传、模型/存储配置、问答聊天记录展示
- ✅ 支持 OpenAI 接口格式的模型配置：`api_key`、`base_url`、`model`、`temperature`。
- ✅ 文档上传后自动切分、写入 Weaviate 与数据库；构建简单知识图谱用于检索。

## 项目结构

```
.
├── app.py                     # Streamlit 应用入口
├── requirements.txt           # 依赖列表
├── multiagent_demo/
│   ├── __init__.py
│   ├── agents.py              # Agent 节点实现（工具调用、Web 搜索等）
│   ├── chunking.py            # 文本切分与关键词提取
│   ├── config.py              # 数据与模型配置数据类
│   ├── database.py            # PostgreSQL/SQLite 操作封装与知识图谱存储
│   ├── graph.py               # LangGraph 工作流编排
│   ├── llm.py                 # 创建 LLM 与 fallback 逻辑
│   └── weaviate_client.py     # Weaviate 客户端封装（含离线兜底检索）
└── README.md
```

## 快速开始

1. **准备环境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **配置环境变量（可选）**
   - 若使用 Weaviate SaaS / OpenAI，可在 `.env` 中配置：
     ```env
     OPENAI_API_KEY=sk-...
     OPENAI_BASE_URL=https://api.openai.com/v1
     WEAVIATE_URL=https://your-instance.weaviate.network
     WEAVIATE_API_KEY=...
     DATABASE_URL=postgresql+psycopg2://user:password@host:port/dbname
     ```
   - 未提供时，可在应用侧边栏直接填写。数据库未配置时默认使用本地 `sqlite:///demo.db`。

3. **运行 demo**
   ```bash
   streamlit run app.py
   ```

4. **在浏览器中体验**
   - 左侧栏填写模型、Weaviate、数据库等连接信息。
   - 上传 TXT/Markdown 等文本文件并点击“写入知识库”。
   - 在主区域输入问题并运行多 Agent 编排，查看向量/关键词/知识图谱/工具/Web 搜索等中间结果以及最终回答。

## 数据流说明

1. **文件上传**：读取文本并根据长度切分为多个 chunk，提取关键词作为知识图谱三元组（`关键词 --appears_in--> 文档`）。
2. **存储**：
   - 向 Weaviate 写入 chunk（若无法连接则自动回退为内存向量检索）。
   - 向 PostgreSQL/SQLite 写入文档、chunk 以及三元组。
3. **多 Agent 流程**：
   - LLM 改写问题，使其更适合检索。
   - 使用改写/原始问题进行向量检索和关键词检索。
   - 基于知识三元组进行关联查询。
   - 根据问题模式触发工具（示例：数学表达式计算）。
   - 调用 DuckDuckGo 完成 Web 搜索补充信息。
   - 汇总上下文，调用 LLM 给出最终回答。

## 注意事项

- **Weaviate**：示例期望已部署 Weaviate（可使用 Weaviate Cloud 或本地 docker）。未连接成功时会自动使用基于 `difflib` 的简易相似度检索。
- **PostgreSQL**：推荐连接真实 PostgreSQL；为便于试用，默认回落到 SQLite 文件 `demo.db`。
- **OpenAI 兼容模型**：只要遵循 OpenAI Chat Completion API 格式即可（如 `Azure OpenAI`、`OpenAI API`、`vLLM` 等）。
- **网络访问**：Web 搜索依赖 DuckDuckGo，如运行环境无外网，应用会给出友好提示并继续执行其他步骤。

## 后续扩展建议

- 引入更强的文本嵌入（如 OpenAI Embeddings / SentenceTransformers）替换简易关键词提取。
- 在 PostgreSQL 中构建真实的知识图谱（图数据库或三元组存储）。
- 拓展工具集，例如实时天气、内部 API、计算服务等。
- 接入 LangSmith 或其他可观测性平台追踪 Agent 调用链路。

祝使用愉快！🚀
