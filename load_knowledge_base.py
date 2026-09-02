"""将 data/knowledge_base.json 中的记录加载到 ChromaDB 向量库。

使用示例：
    python download_model.py          # 第一步：下载本地嵌入模型（仅需执行一次,已下载可跳过）
    python load_knowledge_base.py     # 第二步：构建向量库
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 控制台输出编码容错：防止打印含特殊字符内容时报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 配置（所有路径与密钥均从 .env 读取，遵循 AGENTS.md 规范）
load_dotenv(BASE_DIR / ".env")

# ---------- 配置（均从 .env 读取，代码中仅保留兜底默认值） ----------
# 嵌入模型（AGENTS.md 指定）
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# 本地模型目录（相对项目根目录，目录名取模型 ID 的最后一段）
MODEL_DIR = BASE_DIR / "models" / EMBEDDING_MODEL.split("/")[-1]
# ChromaDB 持久化目录（相对项目根目录，默认 ./chroma_db）
CHROMA_DB_DIR = BASE_DIR / os.getenv("CHROMA_DB_DIR", "chroma_db")
# ChromaDB 集合名称
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gongkao_docs")
# 知识库文件路径
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"

# 每批写入向量库的文档数量
BATCH_SIZE = 64


def add_batch(collection, embedder, ids, documents, metadatas) -> None:
    """将一批文档向量化后写入 ChromaDB 集合。"""
    embeddings = embedder.encode(documents).tolist()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def main() -> None:
    # 强制离线，禁止联网检查
    os.environ["HF_HUB_OFFLINE"] = "1"
    # 检查本地模型是否存在（缺失则提示先运行 download_model.py 下载）
    if not MODEL_DIR.is_dir():
        print("[错误] 未找到本地模型，请先运行 `python download_model.py` 下载模型。")
        raise SystemExit(1)

    if not KNOWLEDGE_BASE_PATH.is_file():
        print(f"[错误] 未找到知识库文件：{KNOWLEDGE_BASE_PATH}")
        raise SystemExit(1)

    # 读取知识库
    records = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    total = len(records)
    print(f"共读取 {total} 条记录，开始加载...")

    # 初始化 sentence-transformers 嵌入模型（从本地路径加载，不联网）
    print(f"正在加载嵌入模型：{MODEL_DIR} ...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(str(MODEL_DIR))

    # 初始化 ChromaDB 持久化客户端
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # 集合已存在则先删除，避免数据重复
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"已删除旧集合：{CHROMA_COLLECTION}")
    except Exception:
        pass  # 集合不存在时忽略

    # 创建集合，使用余弦相似度作为距离度量
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"已创建集合：{CHROMA_COLLECTION}（距离度量：cosine）")

    # 分批收集与写入
    ctx_ids, ctx_docs, ctx_metas = [], [], []
    ana_ids, ana_docs, ana_metas = [], [], []
    context_count = 0
    analysis_count = 0

    for idx, record in enumerate(records, 1):
        source_file = record.get("source_file", "")
        question = record.get("question", "")
        answer = record.get("answer", "")
        context = record.get("context", "")
        analysis = record.get("analysis", "")

        # context 作为 Document 存入，元数据包含 source_file / question / answer
        print(f"正在加载第 {idx}/{total} 条 context...")
        ctx_ids.append(f"ctx-{idx}")
        ctx_docs.append(context)
        ctx_metas.append(
            {
                "source_file": source_file,
                "question": question,
                "answer": answer,
            }
        )
        context_count += 1
        if len(ctx_ids) >= BATCH_SIZE:
            add_batch(collection, embedder, ctx_ids, ctx_docs, ctx_metas)
            ctx_ids, ctx_docs, ctx_metas = [], [], []

        # analysis 作为独立 Document 存入，元数据额外标记 doc_type；空字符串跳过
        if analysis.strip():
            print(f"正在加载第 {idx}/{total} 条 analysis...")
            ana_ids.append(f"ana-{idx}")
            ana_docs.append(analysis)
            ana_metas.append(
                {
                    "doc_type": "analysis",
                    "source_file": source_file,
                    "question": question,
                }
            )
            analysis_count += 1
            if len(ana_ids) >= BATCH_SIZE:
                add_batch(collection, embedder, ana_ids, ana_docs, ana_metas)
                ana_ids, ana_docs, ana_metas = [], [], []

    # 写入剩余批次
    if ctx_ids:
        add_batch(collection, embedder, ctx_ids, ctx_docs, ctx_metas)
    if ana_ids:
        add_batch(collection, embedder, ana_ids, ana_docs, ana_metas)

    # 统计信息
    print(f"共加载 context 文档：{context_count} 条")
    print(f"共加载 analysis 文档：{analysis_count} 条")
    print(f"向量库持久化路径：{CHROMA_DB_DIR}")
    print(f"集合中实际文档数：{collection.count()} 条")


if __name__ == "__main__":
    main()