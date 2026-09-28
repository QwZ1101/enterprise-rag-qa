# 企业知识库 RAG：金融研报问答系统

基于开源 RAG Challenge 冠军方案（[IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2)）改造的本地企业知识库问答项目：以中芯国际相关研报/财报 PDF 为知识库，支持 **Web 界面问答** 与 **命令行问答**，答案附带 PDF 物理页码出处。

## 技术栈

- **PDF 解析**：MinerU 在线 API（200+ 页长 PDF 按页分段解析后合并，见 `scripts/build_content_list.py`）
- **分块**：按真实物理页聚合 + token 切分（`src/text_splitter.py`）
- **检索**：FAISS 向量检索 + BM25 混合检索 + LLM 重排（`src/retrieval.py`、`src/reranking.py`）
- **向量模型**：`qwen3.7-text-embedding-flash`（阿里云 DashScope，默认 1024 维）
- **对话模型**：`qwen3.8-flash`（结构化输出 + 思维链，附页码引用）
- **前端**：Vue 3 单页（`web/`），由 FastAPI 同源托管（`server.py`）

## 目录结构

```
RAG-cy/
├── ask.py            # 命令行问答入口（单问 / --batch 批量）
├── server.py         # Web 服务（FastAPI + 静态前端）
├── main.py           # 上游 CLI（Docling 解析等历史流程，可选）
├── setup.py          # pip install -e . 用
├── requirements.txt
├── .env.example      # API key 模板（复制为 .env 并填密钥；.env 不提交）
├── scripts/
│   ├── build_content_list.py   # PDF -> 带页码 content_list JSON（MinerU 分段）
│   └── rebuild_vector_db.py    # 由已有分块轻量重建 FAISS 向量库
├── src/              # 核心模块（pipeline / 解析 / 分块 / 检索 / 重排 / 问答）
├── web/              # Vue3 前端（index.html + vendor）
├── docs/             # 模块说明、UI 参考图
└── data/stock_data/  # 知识库数据（见下）
```

`data/stock_data/` 内容：

| 路径 | 说明 |
|---|---|
| `pdf_reports/` | 原始研报/财报 PDF |
| `pdf_reports.xlsx` / `subset.csv` | 报告清单（sha1 -> 文件名 -> 公司名） |
| `questions.json` | 问题集（批量问答输入） |
| `debug_data/02_content_list/` | MinerU 解析出的带 0-based `page_idx` 的元素 JSON |
| `debug_data/03_reports_markdown/` | 各报告合并后的 Markdown |
| `databases/chunked_reports/` | 分块结果（块内记录 1-based 物理页） |
| `databases/vector_dbs/` | FAISS 向量库（按报告 sha1 命名） |
| `answers_qwen_turbo*.json` | 批量问答输出结果 |

## 环境准备

要求 Python 3.9（本项目在 3.9 下验证），项目根目录执行：

```powershell
pip install -r requirements.txt
```

将 `.env.example` 复制为 `.env`，填入密钥（`.env` 已被 git 忽略）。最少需要：

```bash
# 问答（对话 + 向量）必需：阿里云 DashScope
DASHSCOPE_API_KEY=sk-xxxx
# 建库 / PDF 解析时才需要：MinerU
MINERU_API_KEY=xxxx
```

也可以直接在系统环境变量里设置同名变量，效果相同（Jina/Gemini/OpenAI 为上游可选，可留空）。

## 建库流程（PDF -> 向量库）

> 仓库**不含**原始 PDF 与 faiss 向量库（`pdf_reports/`、`databases/`、`debug_data/` 已在 `.gitignore` 排除）。clone 后请把自己的 PDF 放进 `data/stock_data/pdf_reports/`，并在 `subset.csv` 登记（`sha1, file_name, company_name`），再按下面三步建库。

```powershell
# 1. MinerU 分段解析 PDF，生成带页码的 content_list（输出到 debug_data/）
python scripts/build_content_list.py

# 2. 按页聚合 + token 分块（输出到 databases/chunked_reports/）
python -c "from pathlib import Path; from src.pipeline import Pipeline; Pipeline(Path('data/stock_data')).chunk_reports()"

# 3. 向量化建库（输出到 databases/vector_dbs/；换 embedding 模型后也用它重建）
python scripts/rebuild_vector_db.py
```

> 若仓库已带 `databases/` 产物，可跳过本节直接问答。

## 问答

### Web 界面

```powershell
python server.py            # 启动并自动打开 http://127.0.0.1:8000
python server.py --port 8010 --no-open
```

接口：`GET /api/health`、`POST /api/ask`（`{"question": "...", "kind": "string|number|boolean|names"}`）、`GET /api/pdf/{sha1}`（浏览器内嵌查看原文 PDF 并跳页）。

### 命令行

```powershell
python ask.py "中芯国际的总部位于哪个城市？"
python ask.py "中芯国际2024年度的营业收入是多少人民币？" --kind number
python ask.py "中芯国际2024年是否盈利？" --kind boolean
python ask.py --batch        # 批量跑 data/stock_data/questions.json
```

打印的页码为 1-based 的 PDF 物理页（与 PDF 阅读器一致）。

## 注意事项

- DashScope 有 QPM/TPM 限流：`server.py` 用锁串行处理请求；批量跑 `--batch` 时勿并发多个进程。
- 问题中需包含公司名（如"中芯国际"）；知识库只有一家公司时 Web 端会自动补全。
- `main.py` / `src/pdf_parsing.py`（Docling 本地解析）、IBM/Gemini 相关代码为上游遗留，当前 MinerU 流程不依赖它们。

## License

MIT（见 `LICENSE`，沿用上游项目）。
