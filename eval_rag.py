# -*- coding: utf-8 -*-
"""评估脚本：在真实 RAG 链路上跑 30 道抽样题，统计来源命中/要点覆盖/耗时。

口径说明：
- 评估集：从知识库 200 条中按 seed=42 分层抽样（单一题 15 + 综合题 15）。
  文章写作/公文写作类参考答案为长文，不适合"要点覆盖率"客观口径，故不纳入本次量化评估。
- 来源命中：返回的 sources（TOP_K=3）中是否包含正确 source_file。
- 要点覆盖：LLM（DeepSeek）将模型回答与参考答案对比，输出 0-100 覆盖率。
- 结果增量写入 eval_results.jsonl，支持断点续跑。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OUT_JSONL = BASE_DIR / "eval_results.jsonl"
SAMPLE_SIZE = 30
SEED = 42

import random
from collections import Counter

from langchain_openai import ChatOpenAI

from app.graph import compiled_graph

def qtype(source_file: str) -> str:
    for t in ["公文写作题", "单一题", "文章写作题", "综合题"]:
        if t in source_file:
            return t
    return "其他"

def load_done() -> set:
    done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["question"])
            except Exception:
                continue
    return done

def call_llm(client: ChatOpenAI, prompt: str, timeout: int = 90) -> str:
    resp = client.invoke(prompt, timeout=timeout)
    return resp.content if hasattr(resp, "content") else str(resp)

def parse_coverage(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {
                "coverage": int(d.get("coverage", -1)),
                "missing": str(d.get("missing", "")),
                "comment": str(d.get("comment", "")),
            }
        except Exception:
            pass
    return {"coverage": -1, "missing": "", "comment": "解析失败"}

JUDGE_TEMPLATE = (
    "你是知识问答评分员。请判断「模型回答」对「参考答案」核心要点的覆盖程度。\n"
    "问题：{question}\n"
    "参考答案（节选）：{reference}\n"
    "模型回答：{answer}\n"
    "评分规则：\n"
    "1. 逐要点对比，模型回答覆盖了参考答案的某个核心要点则计入覆盖；\n"
    "2. 模型回答与要点不符、编造或明显错误，该要点计为未覆盖；\n"
    "3. 若模型基于材料正确作答但与参考答案表述不同，只要要点一致即算覆盖。\n"
    "只输出 JSON，不要输出其他内容："
    '{{"coverage": 0到100的整数, "missing": "缺失或答错的要点简述，无则写无", "comment": "一句话评价"}}'
)

def main() -> None:
    with open(BASE_DIR / "data" / "knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)

    rng = random.Random(SEED)
    pool = [x for x in kb if qtype(x["source_file"]) in ("单一题", "综合题")]
    by_type = {}
    for x in pool:
        by_type.setdefault(qtype(x["source_file"]), []).append(x)
    sample = []
    for t, n in [("单一题", 15), ("综合题", 15)]:
        sample.extend(rng.sample(by_type[t], n))
    rng.shuffle(sample)
    print(f"抽样 {len(sample)} 题：单一题 {sum(1 for x in sample if qtype(x['source_file'])=='单一题')}，综合题 {sum(1 for x in sample if qtype(x['source_file'])=='综合题')}")

    done = load_done()
    client = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.0,
        timeout=90,
        max_retries=1,
    )

    stats = {"done": 0, "skip": 0}
    for i, rec in enumerate(sample, 1):
        q = rec["question"]
        if q in done:
            stats["skip"] += 1
            continue
        t0 = time.time()
        try:
            result = compiled_graph.invoke({"messages": [q]})
        except Exception as exc:
            print(f"[{i}/{len(sample)}] 生成失败: {q[:40]}... -> {exc}")
            row = {"question": q, "error": str(exc), "latency": round(time.time() - t0, 1)}
        else:
            answer = ""
            for m in reversed(result.get("messages", [])):
                c = getattr(m, "content", None)
                if c:
                    answer = str(c)
                    break
            sources = result.get("sources", []) or []
            src_files = [s.get("source_file", "") for s in sources]
            expected = rec["source_file"]
            hit1 = bool(src_files and src_files[0] == expected)
            hit3 = expected in src_files
            nonempty = bool(answer.strip())
            latency = round(time.time() - t0, 1)

            judge = {"coverage": -1, "missing": "", "comment": "未评分"}
            try:
                judge_prompt = JUDGE_TEMPLATE.format(
                    question=q[:300],
                    reference=(rec.get("answer", "") or "")[:1500],
                    answer=answer[:2000],
                )
                judge_text = call_llm(client, judge_prompt)
                judge = parse_coverage(judge_text)
            except Exception as exc:
                print(f"  [评分失败] {exc}")

            row = {
                "question": q,
                "type": qtype(rec["source_file"]),
                "expected_source": expected,
                "hit_source_top1": hit1,
                "hit_source_top3": hit3,
                "answer_nonempty": nonempty,
                "answer": answer,
                "sources": sources,
                "latency": latency,
                "coverage": judge["coverage"],
                "missing": judge["missing"],
                "comment": judge["comment"],
            }
            stats["done"] += 1
            print(f"[{i}/{len(sample)}] {row['type']} coverage={judge['coverage']}% "
                  f"top3_hit={hit3} 耗时={latency}s  | {q[:36]}...")

        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n本次新增 {stats['done']} 题，跳过已完成 {stats['skip']} 题")
    print(f"结果已写入 {OUT_JSONL}")

if __name__ == "__main__":
    main()
