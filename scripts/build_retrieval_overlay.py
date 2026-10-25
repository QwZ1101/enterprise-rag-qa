"""Repair empty table continuation pages from local PDF text; no network calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval_overlay import digest
from src.text_splitter import TextSplitter


def continuation_groups(elements):
    groups = []
    current = None
    for item in elements:
        if item.get('type') != 'table' or type(item.get('page_idx')) is not int:
            continue
        page = item['page_idx'] + 1
        if item.get('table_body') or item.get('text'):
            current = [page]
        elif current and page == current[-1] + 1:
            current.append(page)
            if len(current) == 2:
                groups.append(current)
        else:
            current = None
    # Two tables can share pages; merge overlapping page groups deterministically.
    merged = []
    for group in groups:
        if merged and min(group) <= max(merged[-1]):
            merged[-1] = sorted(set(merged[-1]) | set(group))
        else:
            merged.append(group)
    return merged


def build(corpus, content_list, pdf_path, output):
    import fitz
    corpus, content_list, pdf_path, output = map(Path, (corpus, content_list, pdf_path, output))
    if output.exists():
        raise ValueError('输出已存在；请保留旧版并显式选择新的输出位置')
    document = json.loads(corpus.read_text(encoding='utf-8'))
    elements = json.loads(content_list.read_text(encoding='utf-8'))
    if isinstance(elements, dict):
        elements = elements['content_list']
    groups = continuation_groups(elements)
    if not groups:
        raise ValueError('没有找到可定位的跨页空表续页')
    splitter = TextSplitter()
    pages, chunks = [], []
    with fitz.open(pdf_path) as pdf:
        page_count = len(pdf)
        for group in groups:
            texts = {}
            for number in group:
                if not 1 <= number <= page_count:
                    raise ValueError('解析页码超出 PDF 范围')
                text = pdf[number - 1].get_text('text', sort=True).strip()
                if len(text) < 80:
                    raise ValueError(f'PDF 第 {number} 页文本不足，需 OCR/人工核对，不自动补写')
                texts[number] = text
                pages.append({'page': number, 'text': text, 'source': 'pdf_physical_page'})
            for number in group:
                for index, chunk in enumerate(splitter._split_page({'page': number, 'text': texts[number]})):
                    # The opening page supplies table title/units, explicitly cited separately.
                    chunk.update(id=f'physical:{number}:{index}', source_pages=[number],
                                 provenance='pdf_physical_page', context_pages=[])
                    if number != group[0]:
                        chunk['context_pages'] = [{'page': group[0], 'text': texts[group[0]][:600],
                                                   'role': 'table_opening_context'}]
                    chunks.append(chunk)
    data = dict(version=1, report_id=document['metainfo']['sha1'],
                corpus_sha256=digest(corpus), pdf_sha256=digest(pdf_path),
                content_list_sha256=digest(content_list), pdf_file_name=pdf_path.name,
                pdf_page_count=page_count, groups=groups, replaced_pages=sorted({p for g in groups for p in g}),
                extraction='PyMuPDF get_text(text, sort=True); not OCR or a table-accuracy claim',
                vector_policy='exclude old affected chunks; repaired chunks use lexical retrieval until re-embedded',
                pages=pages, chunks=chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'content-list', 'pdf', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    data = build(args.corpus, args.content_list, args.pdf, args.output)
    print(json.dumps({'groups': data['groups'], 'pages': len(data['pages']), 'chunks': len(data['chunks']),
                      'external_calls': 0, 'output': args.output}, ensure_ascii=False))
