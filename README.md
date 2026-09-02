# RAG 知识库问答 API 原型

基于 **FastAPI + LangGraph** 的知识库问答 API 原型：从公开题目语料中自动提取题干与给定材料，构建本地向量知识库，通过「检索 → 生成」的 RAG 流程，让大模型基于给定材料回答问题并返回引用来源。示例语料使用第三方公考题目（该数据源无开源许可、仅供个人学习、不得商用，版权归原始来源），本项目仅作技术演示，不附带任何题目数据。

---

## 1. 项目简介

- 输入：用户用自然语言就知识库内题目提问。
- 处理：问题经本地 `Sentence-Transformers` 模型编码后，从 ChromaDB 向量库检索最相关的 TOP_K 个文档块，拼接为上下文。
- 输出：DeepSeek（或 OpenAI）大模型基于上下文生成答案，同时返回引用来源（`source_file` 与对应 `question`）。
- 特点：嵌入模型完全本地化（`./models/`），推理过程不依赖 Hugging Face 网络；向量库与知识库均为本地文件，离线可跑。
- 说明：题库数据版权归原始来源，本仓库不附带题目与知识库文件（见 `.gitignore`），请按第 4 节准备自己的题库数据。

---

## 2. 技术架构

| 组件 | 技术选型 | 说明 |
| --- | --- | --- |
| Web 框架 | FastAPI + Uvicorn | 异步 Web 服务，提供 `/chat`、`/health` 等接口，自动生成 Swagger 文档（`/docs`） |
| 编排框架 | LangGraph + LangChain | 以「检索节点 → 生成节点」构建有状态 RAG 流程，导出 `compiled_graph` |
| 向量数据库 | ChromaDB（持久化模式） | 数据保存在 `./chroma_db`，使用余弦相似度（COSINE）度量 |
| 嵌入模型 | Sentence-Transformers | `paraphrase-multilingual-MiniLM-L12-v2`，从本地 `./models/` 离线加载，中英双语效果好 |
| LLM | DeepSeek API（默认）/ OpenAI | 通过 `langchain-openai` 的 OpenAI 兼容接口调用，支持 `timeout=60` |
| 配置管理 | python-dotenv | 所有密钥与路径从 `.env` 读取 |
| 文档解析 | pypdf | 预留 PDF 读取能力（`requirements.txt` 已包含） |

**RAG 流程**：`retrieve_node`（问题编码 → ChromaDB 检索 TOP_K 块 → 拼接 context 与 sources）→ `generate_node`（基于 context 提示词调用 LLM → 生成答案）。

---

## 3. 项目结构

```text
rag-knowledge-qa/
├── app/                          # 应用主代码
│   ├── __init__.py
│   ├── main.py                   # FastAPI 入口（/chat、/health、/）
│   ├── graph.py                  # LangGraph RAG 流程（retrieve → generate）
│   └── models.py                 # Pydantic 请求/响应模型
├── data/
│   └── knowledge_base.json       # 提取生成的知识库（extract 脚本输出）
├── models/
│   └── paraphrase-multilingual-MiniLM-L12-v2/   # 本地嵌入模型（离线加载）
├── chroma_db/                    # ChromaDB 向量库持久化目录（自动生成）
│   ├── chroma.sqlite3
│   └── <collection-uuid>/
├── gongkao-tiku/                 # 题目语料数据源（外部克隆仓库，非本项目代码）
│   ├── 申论题库/                  # 公文写作题 / 单一题 / 文章写作题 / 综合题 等
│   ├── 智能系统/                  # 暂时没有用到
│   ├── 行测题库/                  # 暂无数据
│   ├── 职测题库/                  # 暂无数据
│   └── 综应题库/                  # 暂时没有用到
├── extract_knowledge_data.py     # 提取题目语料 → data/knowledge_base.json
├── download_model.py            # 下载本地嵌入模型（仅需执行一次，已下载可跳过）
├── load_knowledge_base.py        # 加载知识库 → ChromaDB 向量库
├── check_chroma.py               # 检查向量库集合内容
├── requirements.txt              # 依赖清单（锁定主版本号）
├── .env.example                  # 环境变量模板（.env 不入库）
├── .gitignore                    # 排除 .env / 模型 / 题库 / 向量库等
└── AGENTS.md                     # AI 编码约束
```

> 说明：`gongkao-tiku/` 为外部题库仓库，仅作为数据源；`chroma_db/`、`models/`、`data/knowledge_base.json`、`.env` 均不入库（见 `.gitignore`），由脚本自动生成或按需准备。题库内容版权归原始来源，请勿公开传播。

---

## 4. 快速启动

### 4.1 创建虚拟环境并安装依赖（Python 3.11）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填入 LLM 密钥：

```ini
# ===== 检索配置 =====
TOP_K=3
CHROMA_DB_DIR=chroma_db
CHROMA_COLLECTION=gongkao_docs

# ===== LLM 配置（openai / deepseek 二选一，填入对应 API_KEY）=====
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TEMPERATURE=0.2
```

### 4.3 下载本地嵌入模型

```powershell
python download_model.py
```

脚本会检查 `./models/paraphrase-multilingual-MiniLM-L12-v2/` 是否存在：缺失或不完整时，从镜像站（`https://hf-mirror.com`，失败自动重试 5 次）下载完整模型仓库（约 500MB，仅需执行一次）；已存在则直接跳过，不会重复下载。

> **数据准备**：仓库不附带题库与知识库文件（`gongkao-tiku/`、`data/knowledge_base.json` 已 gitignore）。首次使用请自行获取题目数据（可自行获取第三方题库并放到 `./gongkao-tiku`，目录需包含 `申论题库/` 下的 Markdown 题目；注意此类数据仅限个人学习、不得商用），再执行 4.4 提取与 4.5 建库。题库内容版权归原始来源，本仓库仅用于技术演示，请勿公开传播题目原文。

### 4.4 提取题库数据

```powershell
python extract_knowledge_data.py --data_path ./gongkao-tiku
```

每个题型随机提取 50 道（不足则全量），解析 `## 题目`、`## 给定材料`、`## 参考答案`、`## 答题演示`，清洗 Markdown 后输出到 `./data/knowledge_base.json`。

### 4.5 加载向量库

分两步执行：

**第一步：下载本地嵌入模型（仅需执行一次，已下载可跳过）**

```powershell
python download_model.py
```

**第二步：构建向量库**

```powershell
python load_knowledge_base.py
```

将 `context` 与 `analysis` 分别作为文档写入 ChromaDB 集合 `gongkao_docs`（已存在则先清空重建），进度与统计信息实时打印；若本地模型缺失，脚本会提示先运行 `python download_model.py`。

可选：检查向量库：

```powershell
python check_chroma.py
```

### 4.6 启动服务

```powershell
uvicorn app.main:app --reload
```

- 接口文档（Swagger）：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

---

## 5. API 接口文档

### 5.1 `POST /chat` — 问答接口

**请求体**（`QueryRequest`）：

```json
{
  "question": "根据“给定材料”请概括L县“苹果产业后整理”的主要举措。",
  "session_id": "abc-123"
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 用户问题 |
| `session_id` | string | 否 | 会话标识（预留，当前未使用） |

**响应体**（`AnswerResponse`）：

```json
{
  "answer": "根据给定材料，L县“苹果产业后整理”的主要举措有：**政策规划**：……。**个性化包装**：...",
  "sources": [
    {
      "source_file": "单一题/1185_根据给定材料请概括L县苹果产业后整理的主要举措。（对策类）.md",
      "question": "根据“给定材料”请概括L县“苹果产业后整理”的主要举措。"
    },
    {
      "source_file": "单一题/1613_请根据资料，分析说明B市各类行业组织在规范行业行为、推动行业（影响类）.md",
      "question": "请根据资料，分析说明B市各类行业组织在规范行业行为、推动行业发展方面的积极作用。"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `answer` | string | 基于检索材料生成的答案；材料无相关信息时明确返回“不知道” |
| `sources` | array | 引用来源列表，每项含 `source_file`（来源文件）与 `question`（对应题目），按文件去重；无可引用时为 `[]` |

**错误响应**：处理失败时返回 `500`，`detail` 为“处理请求失败，请稍后重试”。

**请求示例**：

```powershell
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"question": "谈谈对基层治理的理解"}'
```

### 5.2 `GET /health` — 健康检查

**响应体**：

```json
{
  "status": "ok"
}
```

### 5.3 `GET /` — 服务说明

返回欢迎信息与接口使用说明；完整交互式文档见 `/docs`。

---

## 6. 效果示例

> 以下为本地真实运行结果（2026-09-02，DeepSeek 生成）：首次问答含本地模型加载约 28s，模型预热后单次问答耗时取决于向量检索与 LLM 生成（约 5–15s）。

**请求**

```json
{
  "question": "请根据给定材料，谈谈如何促进青年返乡创业？"
}
```

**响应**

```json
{
  "answer": "根据给定材料，促进青年返乡创业可以从以下几个方面着手：\n\n1. **提供发展机会**：农村农业、土地等资源丰富，吸引企业落户，增加就业岗位；同时发展农村电商、旅游等产业，为青年提供创新创业机遇。\n2. **完善政策支持**：国家坚持农业农村优先发展，本地给予减免和补贴；放宽基层干部招考门槛，提升晋升空间，吸引人才回流。\n3. **优化人居环境**：整治污染产业，美化村容村貌；完善交通、生活设施及医疗、教育等公共服务，提升乡村宜居性。\n4. **降低生活压力**：农村生活简单、节奏慢，竞争小；房价、物价等生活成本低，生活质量高，增强青年返乡的幸福感。",
  "sources": [
    {
      "source_file": "单一题/760_当下，乡村对年轻人的吸引力正不断增强。请根据给定资料，概括乡（原因类）.md",
      "question": "当下，乡村对年轻人的吸引力正不断增强。请根据“给定资料”，概括乡村对年轻人吸引力提高的原因。"
    },
    {
      "source_file": "文章写作题/700_参考给定资料，结合当前政府推动大众创业，万众创新的社会形势，（文章写作题）.md",
      "question": "参考“给定资料”，结合当前政府推动“大众创业，万众创新”的社会形势，围绕“众筹：金钱之外的价值”，自选角度，自拟题目，写一篇文章。（45分）"
    }
  ]
}
```
### 6.1 量化评估（30 题封闭集回归）

| 指标 | 数值 |
| --- | --- |
| 来源命中（Top-3） | 30/30 = 100% |
| 要点覆盖均值（LLM 评分，人工抽检复核） | 100% |
| 平均单题耗时（模型预热后） | 约 2.6s |

> 口径与局限详见 [评估报告.md](评估报告.md)（真实链路运行，脚本 `eval_rag.py`）。

---

## 7. 测试

项目使用 `pytest + fastapi.testclient` 编写接口测试，测试文件为项目根目录的 `test_api.py`。

### 运行测试

```powershell
.\.venv\Scripts\Activate.ps1
pytest test_api.py -v        # 显示用例结果
pytest test_api.py -v -s     # 额外显示性能测试的打印输出
```

### 通过示例

```text
test_api.py::test_health PASSED
test_api.py::test_chat_success PASSED
test_api.py::test_chat_missing_question PASSED
test_api.py::test_chat_empty_question PASSED
test_api.py::test_chat_performance PASSED
5 passed, 3 warnings in 30.28s
```

> 说明：`3 warnings` 为 `@app.on_event("startup")` 弃用提示（按需求保留该写法）与 `httpx` 库弃用提示，不影响测试结果。

### 性能测试输出示例

```text
[性能] 状态码: 200, 响应时间: 5.4 ms, 响应大小: 167 字节
```

> 注：以上为 `fake_graph` 模式下的 API 层耗时（不含真实检索与 LLM 调用）；接入真实图后响应时间主要取决于向量检索与 LLM 生成耗时。

### 覆盖范围

| 测试用例 | 覆盖内容 |
| --- | --- |
| `test_health` | 健康检查：`GET /health` 返回 `{"status": "ok"}` |
| `test_chat_success` | 正常问答：返回 200，`answer` 非空，`sources` 为含 `source_file` / `question` 的来源列表 |
| `test_chat_missing_question` | 参数校验：缺少 `question` 字段返回 422 |
| `test_chat_empty_question` | 空值处理：空字符串问题仍返回结构化响应 |
| `test_chat_performance` | 性能测试：打印响应时间、状态码、响应体大小 |

### 测试特点

- 通过 `with TestClient(app) as client` 触发 FastAPI `startup` 事件，完整走真实应用生命周期（加载 `compiled_graph`）。
- chat 相关用例默认使用 `fake_graph` fixture（`monkeypatch` 替换 `app.state.graph`，测试结束后自动还原），返回固定的 `answer` 与 `sources`，保证测试离线、确定、不消耗 LLM 配额，同时仍覆盖请求参数校验、响应模型与 `sources` 组装等 API 层逻辑。
- 如需真实端到端测试（真实向量检索 + 真实 LLM），删除/注释对应用例的 `fake_graph` 参数即可，此时需配置好 `.env` 中的 `DEEPSEEK_API_KEY`，且向量库 `./chroma_db` 已加载数据。

### 集成测试（真实链路，单独运行）

`test_integration_rag.py` 走真实 RAG 链路（本地嵌入模型 → ChromaDB 检索 → DeepSeek 生成），验证接口与 LangGraph 图端到端可用；会消耗 LLM 配额，默认被 `pytest.ini` 排除，需显式运行：

```powershell
pytest -m integration -v            # 运行全部集成测试（约 1 分钟，含本地模型加载）
```

缺少 `DEEPSEEK_API_KEY` 或向量库未加载时，用例自动跳过并给出提示。
