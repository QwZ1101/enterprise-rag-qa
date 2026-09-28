# -*- coding: utf-8 -*-
"""
分段调用 MinerU 把整本 PDF 解析为带物理页码的 content_list.json（PDF -> JSON）。

背景：
    MinerU 对 200+ 页的 PDF 一次性解析容易失败（state 很快变 failed，
    err_msg: retry limit reached, please replace the file）；而指定 page_ranges
    只解析若干页则正常。因此这里把整本 PDF 按页分段串行解析，再按各段起始页
    修正 page_idx 偏移后合并成一份完整 content_list。

输出：
    data/stock_data/debug_data/02_content_list/<报告名>.json
    （顶层为元素数组，每个元素含 0-based 物理页 page_idx；供 TextSplitter
      .split_content_list_reports 聚成 1-based 物理页并按 token 分块）
    同时把各段 full.md 顺序合并到 debug_data/03_reports_markdown/<报告名>.md。

依赖：src.pdf_mineru（requests）、PyMuPDF（fitz，用于把原 PDF 无损重存为更兼容的干净 PDF）。

用法（在项目根目录执行）：
    python scripts/build_content_list.py
    python scripts/build_content_list.py --chunk-pages 20
    python scripts/build_content_list.py --pdf "xxx.pdf" --data-root "data/stock_data"
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pdf_mineru  # noqa: E402


def make_clean_pdf(src: Path, dst: Path) -> None:
    """用 PyMuPDF 把原 PDF 无损重存（garbage collect + clean），提升 MinerU 兼容性。"""
    import fitz
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(src)
    doc.save(str(dst), garbage=4, deflate=True, clean=True)
    doc.close()


def _read_segment_result(out_dir: Path):
    """从某段解压结果读取标准 content_list（排除 *_content_list_v2.json）与 full.md。"""
    cands = [p for p in out_dir.rglob("*content_list.json") if "_v2" not in p.name]
    if not cands:
        raise FileNotFoundError("该段结果中未找到 *content_list.json")
    data = json.load(open(cands[0], encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("content_list", [])
    mds = list(out_dir.rglob("full.md"))
    md = mds[0].read_text(encoding="utf-8") if mds else ""
    return items, md


def parse_once(pdf: Path, a: int, b: int, work: Path):
    """解析物理页 [a, b]（1-based，闭区间），返回该段 content_list（page_idx 相对本段从 0 开始）。"""
    page_range = f"{a}-{b}" if a != b else f"{a}"
    out = work / f"seg_{a:04d}_{b:04d}"
    if out.exists():
        shutil.rmtree(out)
    d = pdf_mineru.parse_local_pdf(pdf, out, page_ranges=page_range)
    return _read_segment_result(d)


def parse_range_with_retry(pdf: Path, a: int, b: int, work: Path, min_chunk: int = 5):
    """解析 [a,b]；若整段失败则二分缩小页段递归重试，返回 (items, md)，page_idx 均为相对本段（从 0）。"""
    try:
        return parse_once(pdf, a, b, work)
    except Exception as exc:
        if b - a + 1 <= min_chunk:
            raise RuntimeError(f"页段 {a}-{b} 已缩至最小仍失败: {exc}")
        mid = (a + b) // 2
        print(f"  页段 {a}-{b} 失败（{exc}），二分重试 {a}-{mid} 与 {mid + 1}-{b}")
        left_items, left_md = parse_range_with_retry(pdf, a, mid, work, min_chunk)
        right_items, right_md = parse_range_with_retry(pdf, mid + 1, b, work, min_chunk)
        # 右半段在本段内的起始物理页相对偏移为 (mid+1-a)
        for it in right_items:
            if isinstance(it.get("page_idx"), int):
                it["page_idx"] += (mid + 1 - a)
        return left_items + right_items, "\n\n".join([left_md, right_md]).strip("\n")


def build(pdf_path: Path, data_root: Path, chunk_pages: int = 20, use_clean: bool = True):
    import fitz
    n_pages = fitz.open(pdf_path).page_count
    print(f"原 PDF：{pdf_path.name}，共 {n_pages} 页，每段 {chunk_pages} 页")

    work = Path(tempfile.mkdtemp(prefix="mineru_seg_"))
    upload_pdf = pdf_path
    if use_clean:
        upload_pdf = work / "clean.pdf"
        print("正在生成无损重存的干净 PDF ...")
        make_clean_pdf(pdf_path, upload_pdf)
        print(f"干净 PDF：{upload_pdf} ({upload_pdf.stat().st_size} 字节)")

    # 1-based 闭区间分段
    ranges, start = [], 1
    while start <= n_pages:
        end = min(start + chunk_pages - 1, n_pages)
        ranges.append((start, end))
        start = end + 1

    merged, md_parts = [], []
    for seg_idx, (a, b) in enumerate(ranges, start=1):
        items, md = parse_range_with_retry(upload_pdf, a, b, work)
        offset = a - 1  # 该段相对 page_idx 0 对应全局物理页 a（全局 0-based page_idx = a-1）
        for it in items:
            if isinstance(it.get("page_idx"), int):
                it["page_idx"] += offset
        merged.extend(items)
        if md:
            md_parts.append(md)
        print(f"完成页段 {a}-{b}（第 {seg_idx}/{len(ranges)} 段），本段元素 {len(items)}，累计 {len(merged)}")

    # 写合并后的 content_list
    out_dir = data_root / "debug_data" / "02_content_list"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{pdf_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"已写出 content_list：{out_json}（{out_json.stat().st_size} 字节，{len(merged)} 个元素）")

    # 合并 full.md
    if md_parts:
        md_dir = data_root / "debug_data" / "03_reports_markdown"
        md_dir.mkdir(parents=True, exist_ok=True)
        md_path = md_dir / f"{pdf_path.stem}.md"
        md_path.write_text("\n\n".join(md_parts), encoding="utf-8")
        print(f"已合并 full.md：{md_path}")

    # 诊断信息
    type_counts = {}
    for it in merged:
        type_counts[it.get("type")] = type_counts.get(it.get("type"), 0) + 1
    covered = sorted({it.get("page_idx") for it in merged if isinstance(it.get("page_idx"), int)})
    print("元素类型分布：", type_counts)
    if covered:
        print(f"页码覆盖：物理页 {covered[0] + 1} -> {covered[-1] + 1}，覆盖 {len(covered)}/{n_pages} 页")
    for it in merged:
        if it.get("type") == "table":
            print("表格元素字段样例：", list(it.keys()))
            break
    missing = [p for p in range(n_pages) if p not in set(covered)]
    if missing:
        print("警告：以下 0-based 页未取到任何元素：", missing)

    shutil.rmtree(work, ignore_errors=True)
    print("BUILD_CONTENT_LIST_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",
                    default=str(ROOT / "data" / "stock_data" / "pdf_reports"
                                / "【财报】中芯国际：中芯国际2024年年度报告.pdf"))
    ap.add_argument("--data-root", default=str(ROOT / "data" / "stock_data"))
    ap.add_argument("--chunk-pages", type=int, default=20)
    ap.add_argument("--no-clean", action="store_true", help="不做 PyMuPDF 无损重存，直接上传原 PDF")
    args = ap.parse_args()
    build(Path(args.pdf), Path(args.data_root),
          chunk_pages=args.chunk_pages, use_clean=not args.no_clean)
