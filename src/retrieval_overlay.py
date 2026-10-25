"""Load a versioned physical-page repair without changing frozen vectors/chunks."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_overlay(document_path, document, overlay_dir=None):
    document_path = Path(document_path)
    overlay_dir = Path(overlay_dir) if overlay_dir else document_path.parent.parent / 'retrieval_overlays'
    report_id = str(document['metainfo']['sha1'])
    # Resolve only an immediate child; report identifiers are data, not paths.
    path = overlay_dir / (report_id + '.json')
    if path.resolve().parent != overlay_dir.resolve():
        raise ValueError('Invalid report ID for retrieval overlay')
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('report_id') != report_id:
        raise ValueError('Retrieval overlay version/report mismatch')
    if data.get('corpus_sha256') != digest(document_path):
        raise ValueError('检索修正版与原语料不匹配，请重新构建修正版')
    name = data['pdf_file_name']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Invalid PDF filename in retrieval overlay')
    pdf = document_path.parent.parent.parent / 'pdf_reports' / name
    if not pdf.is_file() or digest(pdf) != data['pdf_sha256']:
        raise ValueError('检索修正版与原 PDF 不匹配，拒绝使用旧页码映射')
    replaced = set(data['replaced_pages'])
    pages = data['pages']
    if (len(replaced) != len(data['replaced_pages']) or not replaced or
            any(type(p) is not int or not 1 <= p <= data['pdf_page_count'] for p in replaced) or
            {p['page'] for p in pages} != replaced or len(pages) != len(replaced)):
        raise ValueError('Invalid physical-page coverage in retrieval overlay')
    texts = {p['page']: p['text'] for p in pages}
    ids = set()
    for chunk in data['chunks']:
        if (chunk['page'] not in replaced or not chunk['text'].strip() or
                chunk['text'] not in texts[chunk['page']] or chunk['id'] in ids):
            raise ValueError('Invalid physical-page chunk in retrieval overlay')
        ids.add(chunk['id'])
        for context in chunk.get('context_pages', []):
            if context['page'] not in replaced or context['text'] not in texts[context['page']]:
                raise ValueError('Invalid cross-page header provenance')
    if {c['page'] for c in data['chunks']} != replaced:
        raise ValueError('Physical-page chunks do not cover every replaced page')
    return data
