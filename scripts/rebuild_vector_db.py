# -*- coding: utf-8 -*-
"""
轻量重建 FAISS 向量库（不依赖 Docling，无需运行完整 pipeline）。

输入：data/stock_data/databases/chunked_reports/*.json（MinerU 转 Markdown 后已分块的结果）
输出：data/stock_data/databases/vector_dbs/<sha1>.faiss
向量模型：qwen3.7-text-embedding-flash（默认 1024 维）

适用场景：更换 embedding 模型后（向量维度变化），用已有分块重新建库。
用法（在项目根目录执行）：
    python scripts/rebuild_vector_db.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import VectorDBIngestor  # noqa: E402


def main():
    root = ROOT / "data" / "stock_data"
    chunked_dir = root / "databases" / "chunked_reports"
    vector_dir = root / "databases" / "vector_dbs"

    if not chunked_dir.exists():
        raise SystemExit(f"未找到分块目录：{chunked_dir}\n请先完成 Markdown 分块（chunk_reports）。")

    vector_dir.mkdir(parents=True, exist_ok=True)
    print(f"[重建] 分块目录：{chunked_dir}")
    print(f"[重建] 输出目录：{vector_dir}")

    VectorDBIngestor().process_reports(chunked_dir, vector_dir)

    print("[重建] 向量库重建完成。")


if __name__ == "__main__":
    main()
