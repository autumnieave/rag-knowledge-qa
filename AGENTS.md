# 项目开发规则（AI 编码约束）

## 1. 技术栈与版本
- Python 3.10+
- Web框架：FastAPI（异步）
- RAG框架：LangGraph + LangChain
- 向量库：ChromaDB（持久化模式，使用本地目录 `./chroma_db`）
- Embedding模型：优先使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（本地/免费），如需使用 OpenAI 则必须从 .env 读取密钥。

## 2. 编码强制规范（防坑）
- **严禁硬编码**：所有 API_KEY、文件路径必须通过 `.env` 文件加载。
- **路径规则**：所有文件读写必须使用 `pathlib.Path` 且基于项目根目录（`BASE_DIR = Path(__file__).parent`），禁止使用 `C:/Users/...` 绝对路径。
- **文本切分**：默认使用 `RecursiveCharacterTextSplitter`，`chunk_size=500`，`chunk_overlap=50`。
- **依赖管理**：任何新增库必须同步更新 `requirements.txt`，且锁定主版本号（如 `langgraph>=0.2.0,<0.3.0`）。

## 3. 项目目录结构（AI 生成代码时必须遵循）
project_root/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI 入口
│ ├── graph.py # LangGraph 流程定义
│ └── models.py # Pydantic 请求/响应模型
├── data/ # 存放原始题目语料 PDF/TXT（此处仅放示例文件）
├── chroma_db/ # 向量库持久化目录（自动生成，不手动提交）
├── models/ # 本地嵌入模型（离线加载）
├── .env.example # 环境变量模板
├── requirements.txt
├── extract_knowledge_data.py # 数据提取脚本
├── load_knowledge_base.py    # 数据入库脚本
├── check_chroma.py           # 向量库检查脚本
└── AGENTS.md # 本文件

## 4. 响应语言
- 代码注释使用**中文**。
- 向我解释技术方案时，尽量简洁，并优先提供可一键复制的完整代码块。