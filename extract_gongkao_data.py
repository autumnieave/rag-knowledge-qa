"""从 gongkao-tiku 项目中提取申论题库数据，统一保存为 knowledge_base.json。

使用示例：
    python extract_gongkao_data.py --data_path ./gongkao-tiku
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# 控制台输出编码容错：防止文件名/内容含 emoji 时打印报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录（脚本所在目录，用于保存输出文件）
BASE_DIR = Path(__file__).resolve().parent

# 每个题型最多随机提取的题目数量
SAMPLE_SIZE = 50

# 题型显示名称（按此顺序统计并输出）
QUESTION_TYPES = ["公文作文题", "单一题", "文章作文题", "综合题"]

# 题型显示名称 -> 可能的实际目录名（兼容真实仓库中“公文写作题/文章写作题”等命名差异）
TYPE_FOLDER_ALIASES = {
    "公文作文题": ["公文作文题", "公文写作题"],
    "单一题": ["单一题"],
    "文章作文题": ["文章作文题", "文章写作题"],
    "综合题": ["综合题"],
}

# 材料序号标记：匹配“材料1”“材料1-2”“材料一”“材料 1：”“【材料1】”等
MATERIAL_MARKER_RE = re.compile(
    r"^\s*[【\[（(]?\s*材料\s*[0-9一二三四五六七八九十百]+"
    r"([-~至到][0-9一二三四五六七八九十百]+)?\s*[】\]）)]?\s*[、．.。:：]?\s*"
)


def read_text_robust(file_path: Path) -> str:
    """读取文本文件，优先 UTF-8，失败时回退 GB18030，并容忍 UTF-8 BOM。"""
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="gb18030")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text


def normalize_heading(name: str) -> str:
    """归一化标题名称：去掉 emoji/符号前缀（如“📌 题目”->“题目”）及结尾冒号。"""
    name = re.sub(r"^[\W_]+", "", name)
    return name.strip().rstrip("：:")


def parse_md_sections(text: str) -> dict:
    """按 Markdown 标题（如“## 题目”“## 📌 题目”）切分文档，返回 {标题: 内容}。
    仅一级/二级标题作为章节边界，三级及以下子标题（如答题演示的“### 第一步”）保留为内容。"""
    sections = {}
    current = None
    lines = text.splitlines()
    buffer = []
    for line in lines:
        match = re.match(r"^#{1,2}\s+(.*?)\s*$", line)
        if match:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = normalize_heading(match.group(1))
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def extract_first_h1(text: str) -> str:
    """提取文档第一个一级标题（# 开头）作为题目的兜底来源。"""
    for line in text.splitlines():
        match = re.match(r"^#\s+(.*)$", line)
        if match:
            return clean_markdown(match.group(1))
    return ""


def clean_markdown(text: str) -> str:
    """移除 Markdown 格式符号（如 **、__、[TOC] 等），只保留纯文本。"""
    if not text:
        return ""
    # 目录标记
    text = re.sub(r"\[TOC\]", "", text, flags=re.IGNORECASE)
    # 代码块与行内代码反引号
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("`", "")
    # 加粗 / 斜体 / 删除线标记（***、**、__、~~、*、_）
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{1,3}", "", text)
    text = text.replace("~~", "")
    # 标题符号（行首 #）、引用符号（行首 >）
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # 列表符号（行首 -、*、+）与有序列表序号（如“1. ”“1、”）
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.、]\s+", "", text, flags=re.MULTILINE)
    # 水平分隔线
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # 链接 [文字](url) 与图片 ![文字](url)：保留文字部分
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # 清理行尾空格、多余空行
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def clean_material(text: str) -> str:
    """移除“材料1”“材料2”等序号标记（仅用于“给定材料”），保留其后文本。"""
    lines = [MATERIAL_MARKER_RE.sub("", line) for line in text.splitlines()]
    return "\n".join(lines).strip()


def extract_question_file(file_path: Path, rel_path: str) -> dict:
    """从单个 .md 题目文件中提取题目、给定材料、参考答案、答题演示（缺失部分留空字符串）。"""
    raw_text = read_text_robust(file_path)
    sections = parse_md_sections(raw_text)

    question = clean_markdown(sections.get("题目", ""))
    if not question:
        # 题目章节为空时，回退到文档第一个一级标题
        question = extract_first_h1(raw_text)
    context = clean_material(clean_markdown(sections.get("给定材料", "")))
    answer = clean_markdown(sections.get("参考答案", ""))
    analysis = clean_markdown(sections.get("答题演示", ""))

    return {
        "source_file": rel_path,
        "question": question,
        "context": context,
        "answer": answer,
        "analysis": analysis,
    }


def resolve_type_dir(data_root: Path, display_name: str) -> Path:
    """按别名列表解析题型实际目录，返回第一个存在的目录。"""
    for alias in TYPE_FOLDER_ALIASES[display_name]:
        candidate = data_root / alias
        if candidate.is_dir():
            return candidate
    return data_root / display_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="提取 gongkao-tiku 申论题库数据并保存为 knowledge_base.json"
    )
    parser.add_argument("--data_path", required=True, help="gongkao-tiku 项目根目录")
    args = parser.parse_args()

    data_root = Path(args.data_path).resolve() / "申论题库"
    if not data_root.is_dir():
        print(f"[错误] 未找到数据目录：{data_root}")
        raise SystemExit(1)

    all_records = []
    summary = []  # 每项为 (题型, 实际提取数, 文件总数)

    for qtype in QUESTION_TYPES:
        type_dir = resolve_type_dir(data_root, qtype)
        if not type_dir.is_dir():
            print(f"[警告] 题型目录不存在，跳过：{type_dir}")
            summary.append((qtype, 0, 0))
            continue

        files = sorted(type_dir.rglob("*.md"))
        total = len(files)
        if total == 0:
            print(f"[警告] 题型目录中未找到 .md 文件：{type_dir}")
            summary.append((qtype, 0, 0))
            continue

        # 随机提取：不足 SAMPLE_SIZE 道则全量提取
        sampled = random.sample(files, min(SAMPLE_SIZE, total))
        records = []
        for file_path in sampled:
            rel_path = file_path.relative_to(data_root).as_posix()
            print(f"正在处理 {rel_path} ...")
            try:
                record = extract_question_file(file_path, rel_path)
                records.append(record)
            except Exception as exc:
                print(f"[错误] 处理 {rel_path} 失败：{exc}")
                continue

        all_records.extend(records)
        summary.append((qtype, len(records), total))

    # 打印每个题型的提取数量统计
    for qtype, count, total in summary:
        if 0 < total < SAMPLE_SIZE:
            print(f"{qtype}：提取 {count} 题（不足{SAMPLE_SIZE}，全量提取）")
        else:
            print(f"{qtype}：提取 {count} 题")
    print(f"总计：提取 {len(all_records)} 题")

    # 保存为 knowledge_base.json（UTF-8，保留中文）
    output_path = BASE_DIR / "data" / "knowledge_base.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存 {len(all_records)} 条记录至 ./data/knowledge_base.json")


if __name__ == "__main__":
    main()