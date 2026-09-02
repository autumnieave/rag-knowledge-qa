"""真实链路集成测试（单独运行，默认不执行）。

运行方式（需 .env 已配置 DEEPSEEK_API_KEY，且 chroma_db 已加载数据）：
    pytest -m integration -v                 # 运行全部集成测试
    pytest test_integration_rag.py -m integration -v

说明：
- 与 test_api.py 的 fake_graph 测试不同，本文件走真实 RAG 链路
  （本地嵌入模型 -> ChromaDB 检索 -> DeepSeek 生成），会消耗 LLM 配额。
- 默认 pytest 通过 pytest.ini 的 addopts 排除 integration 标记，避免误跑。
- 缺少密钥或向量库未加载时自动 skip 并给出提示。
"""

import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import app

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_ready() -> bool:
    """LLM 密钥是否已配置。"""
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _chroma_ready() -> bool:
    """本地向量库是否已加载数据。"""
    import chromadb

    try:
        client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
        collection = client.get_collection(os.getenv("CHROMA_COLLECTION", "gongkao_docs"))
        return collection.count() > 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _env_ready(), reason="未配置 DEEPSEEK_API_KEY，无法调用 LLM"),
    pytest.mark.skipif(not _chroma_ready(), reason="本地向量库未加载，请先运行 load_knowledge_base.py"),
]


@pytest.fixture(scope="module")
def real_client():
    """真实应用客户端：触发 startup 加载真实 compiled_graph。"""
    with TestClient(app) as test_client:
        yield test_client


def test_real_chat_returns_answer_and_sources(real_client):
    """真实链路：POST /chat 应返回非空 answer 与来源列表。"""
    start = time.perf_counter()
    resp = real_client.post("/chat", json={"question": "谈谈对基层治理的理解"})
    elapsed = time.perf_counter() - start
    print(f"[集成] /chat 真实链路耗时: {elapsed:.1f}s")

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"], "真实链路 answer 不应为空"
    assert isinstance(data["sources"], list), "sources 应为列表"
    assert len(data["sources"]) > 0, "真实检索应返回至少一个来源"


def test_real_chat_sources_are_complete(real_client):
    """真实链路：每个来源都应包含 source_file 与 question 字段。"""
    resp = real_client.post("/chat", json={"question": "请简要介绍乡村振兴"})
    assert resp.status_code == 200
    data = resp.json()
    for source in data["sources"]:
        assert source.get("source_file"), "来源缺少 source_file"
        assert source.get("question"), "来源缺少 question"


def test_real_graph_invoke_direct():
    """真实链路：直接调用 LangGraph 图，验证 检索->生成 全链路。"""
    from app.graph import compiled_graph

    result = compiled_graph.invoke({"messages": ["谈谈对基层治理的理解"]})
    answer = ""
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", None)
        if content:
            answer = str(content)
            break
    assert answer, "图调用应生成非空回答"
    assert result.get("sources"), "图调用应返回检索来源"
