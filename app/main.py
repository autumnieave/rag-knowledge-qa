"""FastAPI 主入口（app/main.py）。

运行方式（项目根目录）：
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import AnswerResponse, QueryRequest

# 初始化 FastAPI 应用
app = FastAPI(
    title="公考 RAG 问答接口",
    description="基于 LangGraph + ChromaDB 的公考知识问答服务",
    version="0.1.0",
)

# CORS：允许所有来源（便于本地测试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_graph() -> None:
    """启动时加载编译后的 LangGraph 图，挂载到 app.state 避免每次请求重复加载。"""
    from app.graph import compiled_graph

    app.state.graph = compiled_graph


def _extract_answer(result: dict) -> str:
    """从图返回结果中提取最后一条 AI 消息的内容作为答案。"""
    messages = result.get("messages", [])
    for message in reversed(messages):
        # 兼容 LangChain 消息对象与 dict 两种形态
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content:
            return str(content)
    return ""


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QueryRequest) -> AnswerResponse:
    """接收用户问题，调用 RAG 图生成答案。"""
    try:
        result = app.state.graph.invoke({"messages": [request.question]})
        answer = _extract_answer(result)
        # 从图返回结果中取出检索来源（source_file / question）
        sources = result.get("sources", []) or []
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as exc:
        print(f"[错误] /chat 处理失败：{exc}")
        raise HTTPException(status_code=500, detail="处理请求失败，请稍后重试") from exc


@app.get("/health")
def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    """根路径：返回接口使用说明。"""
    return {
        "message": "欢迎使用公考 RAG 问答接口",
        "docs": "请访问 /docs 查看接口文档",
        "health": "GET /health 健康检查",
        "chat": 'POST /chat 提问（body: {"question": "..."}）',
    }