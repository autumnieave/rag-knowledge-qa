"""LangGraph RAG 流程定义（app/graph.py）。"""

import os
import traceback
from pathlib import Path
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 配置（所有密钥与路径均从 .env 读取，遵循 AGENTS.md 规范）
load_dotenv(BASE_DIR / ".env")

# ---------- 配置（均从 .env 读取，代码中仅保留兜底默认值） ----------
# ChromaDB 持久化目录（相对项目根目录，默认 ./chroma_db）
CHROMA_DB_DIR = BASE_DIR / os.getenv("CHROMA_DB_DIR", "chroma_db")
# ChromaDB 集合名称
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gongkao_docs")
# 检索返回的文档块数量
TOP_K = int(os.getenv("TOP_K", "3"))

# LLM 供应商：openai 或 deepseek
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


class State(TypedDict):
    """LangGraph 图状态：messages 为对话消息列表，context 为检索拼接的上下文，sources 为检索来源。"""

    messages: Annotated[List, add_messages]
    context: str
    sources: List[dict]


def _get_question(state: State) -> str:
    """从状态中取出最近一条用户消息作为当前问题。"""
    last_message = state["messages"][-1]
    return last_message.content if hasattr(last_message, "content") else last_message["content"]


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        print("[DEBUG] 开始加载嵌入模型（本地路径）...")
        from sentence_transformers import SentenceTransformer
        try:
            _embedder = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
            print("[DEBUG] 模型加载成功")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise
    return _embedder


_collection = None


def _get_collection():
    """获取 ChromaDB 集合（持久化目录 ./chroma_db，仅创建一次）。"""
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return _collection


def retrieve_node(state: State) -> dict:
    """检索节点：将问题转为向量，从 ChromaDB 检索 TOP_K 个文档块并拼接为 context。"""
    print("[DEBUG] 进入检索节点")
    question = _get_question(state)

    collection = _get_collection()
    doc_count = collection.count()
    if doc_count == 0:
        # 向量库为空时返回空 context
        return {"context": ""}

    # 使用 sentence-transformers 将问题转为向量
    print("[DEBUG] 开始编码问题...")
    try:
        query_embedding = _get_embedder().encode(question).tolist()
        print("[DEBUG] 编码成功")
    except Exception as e:
        print(f"[ERROR] 编码失败: {e}")
        raise

    # 检索与问题最相似的 TOP_K 个文档块
    n_results = min(TOP_K, doc_count)
    print("[DEBUG] 开始查询向量库...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    print("[DEBUG] 查询完成")

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []

    # 收集检索结果的来源信息（source_file 与 question，按文件去重）
    sources = []
    seen = set()
    for idx, doc in enumerate(documents):
        meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        item = {
            "source_file": meta.get("source_file", ""),
            "question": meta.get("question", ""),
        }
        key = (item["source_file"], item["question"])
        if key not in seen:
            seen.add(key)
            sources.append(item)

    # 将检索结果拼接为 context 字符串（附带来源信息）
    chunks = []
    for idx, doc in enumerate(documents):
        source = ""
        if idx < len(metadatas) and metadatas[idx]:
            source = metadatas[idx].get("source_file", "") or metadatas[idx].get("source", "") or metadatas[idx].get("file_name", "")
        chunks.append(f"[来源：{source}]\n{doc}" if source else doc)
    context = "\n\n".join(chunks)

    return {"context": context, "sources": sources}


SYSTEM_PROMPT_TEMPLATE = """你是一名知识库问答助手。请严格根据以下提供的资料回答问题：

{context}

要求：
1. 只能依据上述资料作答，不要编造资料中不存在的信息。
2. 如果资料中没有与问题相关的信息，请明确回答“不知道”。
3. 回答应简洁、准确。"""


def _get_llm():
    """根据 .env 中的 LLM_PROVIDER 创建 LLM（DeepSeek 使用 ChatOpenAI 的兼容接口）。"""
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置")
        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=LLM_TEMPERATURE,
            timeout=60,
        )

    if not OPENAI_API_KEY:
        raise ValueError("缺少 OPENAI_API_KEY，请在 .env 中配置")
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
        temperature=LLM_TEMPERATURE,
        timeout=60,
    )


def generate_node(state: State) -> dict:
    """生成节点：基于 context 调用 LLM 生成答案，并存入 messages。"""
    print("[DEBUG] 进入生成节点")
    question = _get_question(state)
    context = state.get("context", "")

    # 提示词模板：要求 LLM 严格基于 context 回答，信息不足时明确说“不知道”
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _get_llm()
    print("[DEBUG] 开始调用 LLM...")
    response = chain.invoke({"context": context, "question": question})
    print("[DEBUG] LLM 调用完成")
    answer = response.content if hasattr(response, "content") else str(response)

    return {"messages": [AIMessage(content=answer)], "sources": state.get("sources", [])}


# ---------- 构建并编译图 ----------
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

# 导出编译后的图
compiled_graph = graph.compile()
