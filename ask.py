# -*- coding: utf-8 -*-
"""
命令行问答入口（答案附带 PDF 物理页码出处）。

前提：已完成建库（debug_data/02_content_list 带页码 JSON、databases/chunked_reports 分块、
databases/vector_dbs 向量库）。模型：对话 qwen3.8-flash、向量 qwen3.7-text-embedding-flash。

用法（在项目根目录，用 Python3.9）：
    # 单条提问（默认 string 类型；问题中需包含公司名，如“中芯国际”）
    python ask.py "中芯国际的总部位于哪个城市？"
    python ask.py "中芯国际2024年度的营业收入是多少人民币？" --kind number
    python ask.py "中芯国际2024年是否盈利？" --kind boolean

    # 批量跑 data/stock_data/questions.json，结果写到 data/stock_data/answers*.json
    python ask.py --batch

说明：打印的页码为 1-based 的 PDF 物理页（与 PDF 阅读器/打印页一致）；
批量生成的比赛提交文件内部会按赛事约定把 page_index 转为 0-based。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline, RunConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RAG 问答命令行（带页码出处）")
    ap.add_argument("question", nargs="?", default=None,
                    help="问题文本（需包含公司名，如：中芯国际的总部在哪个城市？）")
    ap.add_argument("--kind", default="string",
                    choices=["string", "number", "boolean", "names"],
                    help="答案类型，默认 string")
    ap.add_argument("--batch", action="store_true",
                    help="批量处理 data/stock_data/questions.json 并生成 answers 文件")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "stock_data"))
    args = ap.parse_args()

    pipeline = Pipeline(Path(args.data_root), run_config=RunConfig())

    if args.batch:
        pipeline.process_questions()
        return

    if not args.question:
        ap.error("请提供问题文本，或使用 --batch 批量处理 questions.json")

    answer = pipeline.answer_single_question(args.question, kind=args.kind)

    print("\n================ 答案 ================")
    print(answer.get("final_answer"))
    pages = sorted({p for p in (answer.get("relevant_pages") or []) if isinstance(p, int)})
    refs = answer.get("references") or []
    if pages:
        print("------------------------------------")
        print("页码出处（PDF 物理页，1-based）：", pages)
    if refs:
        print("references：", json.dumps(refs, ensure_ascii=False))
    print("====================================")


if __name__ == "__main__":
    main()
