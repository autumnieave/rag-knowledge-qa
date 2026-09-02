"""下载本地嵌入模型（paraphrase-multilingual-MiniLM-L12-v2）。

若 ./models/paraphrase-multilingual-MiniLM-L12-v2/ 不存在或不完整，
则使用 huggingface_hub.snapshot_download 从镜像站下载模型仓库
（包含 config.json、model.safetensors、tokenizer.json 等 SentenceTransformer
运行所需的全部文件）；本地模型已存在则直接跳过，不会重复下载。

下载过程更稳健：
- 断点续传（resume_download=True；新版 huggingface_hub 默认即断点续传，
  已下载的部分文件自动保留复用，不重复下载）；
- 下载前检查：本地已有部分文件时保留不删除，仅补齐缺失文件；
- 单线程下载（max_workers=1），减少网络波动影响；
- 仅下载运行必需文件，跳过 onnx / openvino / tf / pytorch 等备用权重
  （避免数 GB 的无效下载，显著降低卡顿概率）；
- 失败自动重试 MAX_RETRIES 次；
- 重试仍失败时，打印手动下载链接与放置路径。

使用示例：
    python download_model.py
"""

import inspect
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 使用国内镜像源下载模型（若该镜像不可用，可替换为以下备选之一）：
# - https://mirror.sjtu.edu.cn/huggingface（上海交大）
# - https://mirrors.tuna.tsinghua.edu.cn/huggingface（清华大学）
# 注意：须早于 huggingface_hub 导入执行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
# 下载失败时的最大重试次数
MAX_RETRIES = 5
# 每次重试前的等待秒数
RETRY_WAIT_SECONDS = 5

# 跳过无需下载的备用权重格式（onnx / openvino / tf / pytorch 等），
# 仅保留 SentenceTransformer 运行所需的文件，避免数 GB 无效下载
IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "pytorch_model.bin",
    "tf_model.h5",
]

# 判断模型仓库是否完整所需的关键文件（缺失任一则视为不完整）
MODEL_REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "1_Pooling/config.json",
]

# 手动下载兜底时需要的关键文件（覆盖运行所需全部文件）
MANUAL_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "README.md",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
]


def is_model_complete() -> bool:
    """判断本地模型目录是否存在且包含完整仓库文件。"""
    return MODEL_DIR.is_dir() and all(
        (MODEL_DIR / filename).is_file() for filename in MODEL_REQUIRED_FILES
    )


def _print_existing_files() -> None:
    """下载前检查：打印本地已存在的部分内容（保留不删除，可断点续传）。"""
    if not MODEL_DIR.is_dir():
        print("本地尚无任何模型文件，将全新下载。")
        return
    files = [p.name for p in MODEL_DIR.iterdir() if p.is_file()]
    dirs = [p.name for p in MODEL_DIR.iterdir() if p.is_dir()]
    print(
        f"本地已存在部分内容：{len(files)} 个文件、{len(dirs)} 个子目录"
        "（保留不删除，仅补齐缺失文件，避免重复下载）。"
    )
    if files:
        print("已存在文件：" + "、".join(sorted(files)))
    missing = [
        name for name in MODEL_REQUIRED_FILES if not (MODEL_DIR / name).is_file()
    ]
    if missing:
        print("仍需补齐（缺失）：" + "、".join(missing))


def _snapshot_download_kwargs() -> dict:
    """构造 snapshot_download 参数，自动兼容不同版本的 huggingface_hub。"""
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": EMBEDDING_MODEL,
        "local_dir": str(MODEL_DIR),  # 完整仓库文件直接写入目标目录（非缓存格式）
        "ignore_patterns": IGNORE_PATTERNS,  # 跳过 onnx/openvino/tf/pytorch 等备用权重
        "max_workers": 1,  # 单线程下载，减少网络波动影响
    }
    # 启用断点续传：旧版 huggingface_hub 需显式传入 resume_download=True；
    # huggingface_hub >= 1.0 已移除该参数（断点续传为默认行为），
    # 此处按版本能力自动适配，避免新版调用报错。
    if "resume_download" in inspect.signature(snapshot_download).parameters:
        kwargs["resume_download"] = True
    return kwargs


def download_model() -> Path:
    """从镜像站下载模型仓库到本地目录（带重试与手动兜底）。

    local_dir 模式会将仓库文件直接写入目标目录（完整文件，而非缓存格式），
    已存在的部分文件不会被删除，snapshot_download 会自动断点续传。
    """
    from huggingface_hub import snapshot_download

    # 下载前检查：打印已存在的部分文件（保留不删除，避免重复下载）
    _print_existing_files()

    kwargs = _snapshot_download_kwargs()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"第 {attempt}/{MAX_RETRIES} 次尝试下载模型：{EMBEDDING_MODEL} ...")
            snapshot_download(**kwargs)
            print(f"模型下载完成：{MODEL_DIR}")
            return MODEL_DIR
        except Exception as e:
            last_error = e
            print(f"[错误] 第 {attempt} 次下载失败：{e}")
            if attempt < MAX_RETRIES:
                print(f"将在 {RETRY_WAIT_SECONDS} 秒后重试 ...")
                time.sleep(RETRY_WAIT_SECONDS)

    # 自动下载失败：提示手动下载（给出具体链接与放置路径）
    print()
    print("[提示] 自动下载多次失败，请手动从镜像站下载模型（二选一）：")
    print(f"  方式一（推荐，整仓下载）：")
    print(f"    git clone https://hf-mirror.com/{EMBEDDING_MODEL} \"{MODEL_DIR}\"")
    print("    说明：需已安装 git-lfs（大文件由 LFS 管理）。")
    print(f"  方式二（逐个文件下载）：")
    print(f"    模型主页：https://hf-mirror.com/{EMBEDDING_MODEL}")
    print(f"    请将下列文件下载到本地目录（目录不存在时先创建）：{MODEL_DIR}")
    for name in MANUAL_FILES:
        print(f"      - https://hf-mirror.com/{EMBEDDING_MODEL}/resolve/main/{name}")
    print("  下载完成后再次运行本脚本，会自动校验完整性。")
    raise SystemExit(1) from last_error


def main() -> None:
    if is_model_complete():
        print(f"本地模型已存在且完整，无需下载：{MODEL_DIR}")
        return

    print(f"本地模型缺失或不完整：{MODEL_DIR}")
    print("开始下载模型仓库（仅运行必需文件，仅需执行一次）...")
    download_model()

    # 下载后校验：仍不完整则给出提示，避免带病使用
    if not is_model_complete():
        print("[警告] 模型文件仍不完整，请按上方提示手动补齐后再运行。")
        raise SystemExit(1)
    print(f"模型就绪：{MODEL_DIR}")


if __name__ == "__main__":
    main()