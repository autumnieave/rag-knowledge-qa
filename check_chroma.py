"""检查 ChromaDB 向量库中集合的内容（check_chroma.py）。

使用示例：
    python check_chroma.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 控制台输出编码容错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 配置（路径与集合名从 .env 读取，遵循 AGENTS.md 规范）
load_dotenv(BASE_DIR / ".env")

# ChromaDB 持久化目录（相对项目根目录，默认 ./chroma_db）
CHROMA_DB_DIR = BASE_DIR / os.getenv("CHROMA_DB_DIR", "chroma_db")
# ChromaDB 集合名称
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gongkao_docs")


def main() -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    print(f"向量库持久化路径：{CHROMA_DB_DIR}")

    # 读取集合；不存在时给出提示
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        print(f"[提示] 集合 {CHROMA_COLLECTION} 不存在，请先运行 load_knowledge_base.py 加载数据。")
        raise SystemExit(1)

    count = collection.count()
    print(f"集合名称：{CHROMA_COLLECTION}")
    print(f"文档总数：{count}")

    # 集合为空时打印提示
    if count == 0:
        print("[提示] 集合为空，请先运行 load_knowledge_base.py 加载数据。")
        return

    # 使用 peek 查看前 5 条记录
    result = collection.peek(limit=5)
    ids = result.get("ids", []) or []
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []

    print(f"\n前 {min(5, count)} 条记录：")
    for idx, doc_id in enumerate(ids):
        doc = documents[idx] if idx < len(documents) else ""
        meta = metadatas[idx] if idx < len(metadatas) else {}
        print("-" * 40)
        print(f"ID：{doc_id}")
        print(f"元数据：{meta}")
        print(f"内容（前200字符）：{doc[:200]}")


if __name__ == "__main__":
    main()