# -*- coding: utf-8 -*-
"""
企业知识库 RAG 的 Web 服务。

- FastAPI 暴露问答接口 POST /api/ask（复用 src.pipeline.Pipeline.answer_single_question）
- 同时静态托管 web/ 目录下的 Vue3 单页前端（同源，无跨域问题）

启动（在项目根目录、使用 Python3.9）：
    python server.py                 # 默认 http://127.0.0.1:8000 ，并自动打开浏览器
    python server.py --port 8010
    python server.py --no-open       # 不自动打开浏览器

接口：
    GET  /api/health          健康检查 / 公司列表 / 模型信息
    POST /api/ask             入参 {"question": str, "kind": str}
                              kind ∈ string | number | boolean | names，默认 string
"""
import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline, RunConfig  # noqa: E402
from trace_logging import traced_request, record_error

DATA_ROOT = ROOT / "data" / "stock_data"
WEB_DIR = ROOT / "web"
VALID_KINDS = ["string", "number", "boolean", "names"]


def _read_subset():
    """读取 subset.csv（utf-8 优先，失败回退 gbk）。"""
    csv_path = DATA_ROOT / "subset.csv"
    for enc in ("utf-8", "gbk"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame(columns=["sha1", "file_name", "company_name"])


SUBSET_DF = _read_subset()
COMPANIES = [str(c) for c in SUBSET_DF["company_name"].dropna().unique().tolist()]
# sha1 -> 报告文件名（不带扩展名），用于定位原始 PDF
SHA1_TO_FILE = {str(r.sha1): str(r.file_name) for r in SUBSET_DF.itertuples()}

app = FastAPI(title="企业知识库 RAG")

# Pipeline 只做路径配置，初始化很轻；检索器在每次问答时按需构建
pipeline = Pipeline(DATA_ROOT, run_config=RunConfig())

# DashScope 有 QPM/TPM 限流，本地单用户用一把锁串行问答，最稳
_ask_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str
    kind: str = "string"
    report_id: Optional[str] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "companies": COMPANIES,
        "chat_model": pipeline.run_config.answering_model,
        "embedding_model": "qwen3.7-text-embedding-flash",
        "retrieval_mode": pipeline.run_config.retrieval_mode,
        "reports": [{'report_id': str(r.sha1), 'company': str(r.company_name), 'title': str(r.file_name)}
                    for r in SUBSET_DF.itertuples()],
    }


@app.post("/api/ask")
@traced_request("rag", ROOT)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    kind = (req.kind or "string").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "问题不能为空"})
    if kind not in VALID_KINDS:
        kind = "string"

    # 公司名兜底：问题里没有任何已知公司名时，单公司知识库自动补上
    hit = [c for c in COMPANIES if c in question]
    if hit:
        company = hit[0]
        q_for_llm = question
    elif len(COMPANIES) == 1:
        company = COMPANIES[0]
        q_for_llm = f"{company}，{question}"
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"问题中需要包含公司名，知识库包含：{ '、'.join(COMPANIES) }"},
        )

    matches = SUBSET_DF[SUBSET_DF['company_name'] == company]
    if req.report_id is not None:
        matches = matches[matches['sha1'].astype(str) == req.report_id]
    if len(matches) != 1:
        return JSONResponse(status_code=400, content={'error': '请明确唯一 report_id，避免混用同公司不同报告。',
                            'reports': [{'report_id': str(r.sha1), 'title': str(r.file_name)}
                                        for r in SUBSET_DF[SUBSET_DF['company_name'] == company].itertuples()]})
    selected_report = str(matches.iloc[0]['sha1'])
    t0 = time.time()
    try:
        with _ask_lock:
            ans = pipeline.answer_single_question(q_for_llm, kind=kind, report_id=selected_report)
    except Exception as e:  # noqa: BLE001 - 把后端异常友好地返回给前端
        record_error(e, handled=True)
        return JSONResponse(status_code=500, content={"error": f"问答失败：{e}"})

    pages = sorted({p for p in (ans.get("relevant_pages") or []) if isinstance(p, int)})
    return {
        "answer": ans.get("final_answer"),
        "kind": kind,
        "company": company,
        "rewritten_question": q_for_llm if q_for_llm != question else question,
        "pages": pages,                                   # 1-based PDF 物理页
        "references": ans.get("references") or [],       # [{pdf_sha1, page_index}]
        "reasoning_summary": ans.get("reasoning_summary"),
        "elapsed_sec": round(time.time() - t0, 2),
        "error": None,
    }


@app.get("/api/pdf/{sha1}")
def get_pdf(sha1: str):
    """按 subset 中的 sha1 返回对应原始 PDF（inline，浏览器内置查看器可用 #page=N 跳页）。"""
    file_name = SHA1_TO_FILE.get(sha1)
    if not file_name:
        return JSONResponse(status_code=404, content={"error": f"未找到 sha1={sha1} 对应的 PDF"})
    pdf_path = DATA_ROOT / "pdf_reports" / f"{file_name}.pdf"
    if not pdf_path.exists():
        return JSONResponse(status_code=404, content={"error": f"PDF 文件不存在：{file_name}.pdf"})
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


# 静态前端兜底路由（必须放在 /api 路由之后注册）
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    import webbrowser

    ap = argparse.ArgumentParser(description="企业知识库 RAG Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"企业知识库 Web 已启动：{url} （Ctrl+C 停止）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
