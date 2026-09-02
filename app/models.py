"""Pydantic 请求/响应模型（遵循 AGENTS.md 目录结构中的 models.py）。"""

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """接收用户提问的请求模型。"""

    question: str
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """返回答案与引用来源的响应模型。"""

    answer: str
    sources: Optional[List[dict]] = None
