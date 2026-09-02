"""接口测试（pytest + fastapi.testclient）。

运行方式：
    .\\.venv\\Scripts\\Activate.ps1
    pytest test_api.py -v            # 显示用例结果
    pytest test_api.py -v -s         # 额外显示性能测试的打印输出

说明：
- 测试通过 `with TestClient(app) as client` 触发 FastAPI startup 事件（加载 compiled_graph）。
- chat 相关用例默认使用固定返回的假图（fake_graph fixture），保证离线、确定、不消耗 LLM 配额；
  如需真实端到端测试，删除/注释对应用例中的 `fake_graph` 参数即可（此时需配置好 .env 中的
  DEEPSEEK_API_KEY，且向量库 ./chroma_db 已加载数据）。
- 运行前请先安装依赖：pip install -r requirements.txt 以及 pip install pytest。
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """创建测试客户端，with 语句触发 startup 事件加载图。"""
    with TestClient(app) as test_client:
        yield test_client


class FakeGraph:
    """模拟 LangGraph compiled_graph 的返回结果，避免测试依赖网络与 LLM。"""

    def invoke(self, state):
        return {
            "messages": [{"role": "assistant", "content": "这是测试答案：基层治理需要多方协同。"}],
            "sources": [
                {"source_file": "单一题/001.md", "question": "请概括基层治理的主要做法。"},
            ],
        }


@pytest.fixture
def fake_graph(client, monkeypatch):
    """将 app.state.graph 替换为假图，测试结束后自动还原。"""
    graph = FakeGraph()
    monkeypatch.setattr(app.state, "graph", graph)
    return graph


def test_health(client):
    """健康检查接口应返回 {"status": "ok"}。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_success(client, fake_graph):
    """正常问答：应返回 200，且 answer 非空、sources 为来源列表。"""
    resp = client.post("/chat", json={"question": "谈谈对基层治理的理解"})
    assert resp.status_code == 200

    data = resp.json()
    assert "answer" in data, "响应缺少 answer 字段"
    assert data["answer"], "answer 不应为空字符串"
    assert "sources" in data, "响应缺少 sources 字段"
    assert isinstance(data["sources"], list), "sources 应为列表"
    assert len(data["sources"]) > 0, "sources 不应为空"
    assert data["sources"][0]["source_file"] == "单一题/001.md", "来源文件不匹配"
    assert data["sources"][0]["question"], "来源题目不应为空"


def test_chat_missing_question(client):
    """缺少 question 字段：FastAPI 参数校验应返回 422。"""
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_empty_question(client, fake_graph):
    """空字符串问题：系统应正常处理并返回结构化响应。"""
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data.get("answer"), str), "answer 应为字符串"
    assert isinstance(data.get("sources"), list), "sources 应为列表"


def test_chat_performance(client, fake_graph):
    """性能测试：记录并打印响应时间、状态码与响应体大小。"""
    start = time.perf_counter()
    resp = client.post("/chat", json={"question": "请简要介绍乡村振兴"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    size = len(resp.content)

    print(f"[性能] 状态码: {resp.status_code}, 响应时间: {elapsed_ms:.1f} ms, 响应大小: {size} 字节")

    assert resp.status_code == 200
    assert resp.json().get("answer"), "answer 不应为空"