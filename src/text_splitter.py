import json
import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

# 文本分块工具类，支持按页分块、表格插入、token统计等
class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """按页分组已序列化表格，便于后续插入到对应页面分块中"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """将报告按页分块，保留markdown表格内容，可选插入序列化表格块。"""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            # 加载序列化表格，按页分组
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            # 普通文本分块
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            # 插入序列化表格分块
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 统计字符串的token数，支持自定义编码
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """将单页文本分块，保留原始markdown表格。"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    #对 json 文件分块，输出还是 json
    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。
        如果提供了序列化表格目录，会尝试将表格内容插入到对应页面的分块中。
        主要用于后续向量化和检索的预处理。
        参数：
            all_report_dir: 存放待处理报告json的目录
            output_dir: 分块后输出的目标目录
            serialized_tables_dir: （可选）存放序列化表格的目录
        """
        # 获取所有报告文件路径
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        # 遍历每个报告文件
        for report_path in all_report_paths:
            serialized_tables_path = None
            # 如果提供了表格序列化目录，查找对应表格文件
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"警告：未找到 {report_path.name} 的序列化表格报告")
                
            # 读取报告内容
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            # 分块处理，插入表格分块（如有）
            updated_report = self._split_report(report_data, serialized_tables_path)
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入分块后的报告到目标目录
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        # 输出处理文件数统计
        print(f"已分块处理 {len(all_report_paths)} 个文件")

    def split_markdown_file(self, md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5):
        """
        按行分割 markdown 文件，每个分块记录起止行号和内容。
        :param md_path: markdown 文件路径
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :return: 分块列表
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        chunks = []
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            start = i
            end = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[start:end])
            chunks.append({
                'lines': [start + 1, end],  # 行号从1开始
                'text': chunk_text
            })
            i += chunk_size - chunk_overlap
        return chunks

    def split_markdown_reports(self, all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None):
        """
        批量处理目录下所有 markdown 文件，分块并输出为 json 文件到目标目录。
        :param all_md_dir: 存放 .md 文件的目录
        :param output_dir: 输出 .json 文件的目录
        :param chunk_size: 每个分块的最大行数
        :param chunk_overlap: 分块重叠行数
        :param subset_csv: subset.csv 路径，用于建立 file_name 到 company_name 的映射
        """
        # 建立 file_name（去扩展名）到 company_name 的映射
        file2company = {}
        file2sha1 = {}
        if subset_csv is not None and os.path.exists(subset_csv):
            # 优先尝试 utf-8，失败则尝试 gbk
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            # 自动识别主键列
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')
        
        all_md_paths = list(all_md_dir.glob("*.md"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for md_path in all_md_paths:
            chunks = self.split_markdown_file(md_path, chunk_size, chunk_overlap)
            output_json_path = output_dir / (md_path.stem + ".json")
            # 查找 company_name 和 sha1
            file_no_ext = md_path.stem
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            # metainfo 只保留 sha1、company_name、file_name 字段
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": md_path.name}
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({"metainfo": metainfo, "content": {"chunks": chunks}}, f, ensure_ascii=False, indent=2)
            print(f"已处理: {md_path.name} -> {output_json_path.name}")
        print(f"共分割 {len(all_md_paths)} 个 markdown 文件")

    # ============ PDF->JSON（MinerU content_list.json，元素带 page_idx）分块 ============
    def _html_table_to_text(self, html: str) -> str:
        """把 MinerU 表格的 HTML（table_body）轻量转成可读文本，避免标签噪声影响向量化。"""
        if not html:
            return ""
        s = re.sub(r"</\s*tr\s*>", "\n", html, flags=re.IGNORECASE)
        s = re.sub(r"</\s*(td|th)\s*>", " | ", s, flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", "", s)
        s = (s.replace("&nbsp;", " ").replace("&#160;", " ")
               .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n\s*\n+", "\n", s)
        return s.strip()

    def split_content_list_file(self, content_list: List[Dict], chunk_size: int = 300, chunk_overlap: int = 50):
        """
        将单个报告的 MinerU content_list（元素数组，每项含 page_idx）聚合为按页文本，再按 token 分块。
        页码口径：MinerU 的 page_idx 为 0-based 物理页，这里统一输出为 1-based 物理页码（page = page_idx + 1）。
        返回 (pages, chunks)：
          pages: [{"page": 页码, "text": 该页全文, "length_tokens": n}]（供父文档检索）
          chunks: [{"id","type":"content","page","length_tokens","text"}]（供向量化/检索）
        """
        page_texts = {}
        # 页眉/页脚/页码是每页重复的装饰元素，不进入检索语料；正文 text、table 等才入库
        skip_types = {"header", "footer", "page_number"}
        for item in content_list:
            page_idx = item.get("page_idx")
            if page_idx is None:
                continue
            if item.get("type") in skip_types:
                continue
            text = item.get("text")
            if not text:
                # 表格元素通常把内容放在 table_body（HTML）
                table_body = item.get("table_body")
                text = self._html_table_to_text(table_body) if table_body else ""
            text = (text or "").strip()
            if not text:
                continue
            page_no = int(page_idx) + 1  # 转 1-based 物理页码
            page_texts.setdefault(page_no, []).append(text)

        pages, chunks = [], []
        chunk_id = 0
        for page_no in sorted(page_texts):
            page_text = "\n\n".join(page_texts[page_no]).strip()
            page_obj = {"page": page_no, "text": page_text, "length_tokens": self.count_tokens(page_text)}
            pages.append(page_obj)
            for chunk in self._split_page(page_obj, chunk_size, chunk_overlap):
                chunk["id"] = chunk_id
                chunk["type"] = "content"
                chunk_id += 1
                chunks.append(chunk)
        return pages, chunks

    def split_content_list_reports(self, all_json_dir: Path, output_dir: Path,
                                   chunk_size: int = 300, chunk_overlap: int = 50, subset_csv: Path = None):
        """
        批量处理 MinerU 的 content_list JSON（PDF->JSON 产物，元素带 page_idx），
        按页分块并输出与既有流程一致的报告 JSON（含 metainfo、content.pages、content.chunks）。
        :param all_json_dir: 存放 *content_list.json（或规范化后的 <报告名>.json，顶层为元素数组）的目录
        :param output_dir: 分块后输出目录（chunked_reports）
        :param subset_csv: subset.csv，用于映射 file_name -> company_name / sha1
        """
        # 建立 file_name（去扩展名）到 company_name / sha1 的映射（与 markdown 流程一致）
        file2company, file2sha1 = {}, {}
        if subset_csv is not None and os.path.exists(subset_csv):
            try:
                df = pd.read_csv(subset_csv, encoding='utf-8')
            except UnicodeDecodeError:
                print('警告：subset.csv 不是 utf-8 编码，自动尝试 gbk 编码...')
                df = pd.read_csv(subset_csv, encoding='gbk')
            if 'file_name' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = os.path.splitext(str(row['file_name']))[0]
                    file2company[file_no_ext] = row['company_name']
                    if 'sha1' in row:
                        file2sha1[file_no_ext] = row['sha1']
            elif 'sha1' in df.columns:
                for _, row in df.iterrows():
                    file_no_ext = str(row['sha1'])
                    file2company[file_no_ext] = row['company_name']
                    file2sha1[file_no_ext] = row['sha1']
            else:
                raise ValueError('subset.csv 缺少 file_name 或 sha1 列，无法建立文件名到公司名的映射')

        all_json_paths = list(Path(all_json_dir).glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for json_path in all_json_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # MinerU content_list.json 顶层为数组；兼容个别封装为 {"content_list": [...]} 的情况
            content_list = data if isinstance(data, list) else data.get("content_list", [])
            pages, chunks = self.split_content_list_file(content_list, chunk_size, chunk_overlap)

            file_no_ext = json_path.stem
            # 去掉 MinerU 常见的 "_content_list" 后缀，使其与报告名 / subset 映射对齐
            if file_no_ext.endswith("_content_list"):
                file_no_ext = file_no_ext[: -len("_content_list")]
            company_name = file2company.get(file_no_ext, "")
            sha1 = file2sha1.get(file_no_ext, "")
            metainfo = {"sha1": sha1, "company_name": company_name, "file_name": file_no_ext + ".pdf"}
            report = {"metainfo": metainfo, "content": {"pages": pages, "chunks": chunks}}
            with open(output_dir / (file_no_ext + ".json"), 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"已处理(JSON带页码): {json_path.name} -> {file_no_ext}.json，页数={len(pages)}，分块={len(chunks)}")
        print(f"共分割 {len(all_json_paths)} 个 content_list JSON 文件")
