"""Offline diagnostic: saved vector top-10 + current BM25. Never calls an API.

The saved pool is smaller than current production retrieval, so this is not a
fresh end-to-end evaluation or a measured improvement of the live system.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics import aggregate_retrieval, digest, load_frozen
from src.fusion_retrieval import FusionRetriever
from src.retrieval import VectorRetriever
from trace_logging import code_version, sanitize


def run(predictions_path, output):
    predictions_path, output = Path(predictions_path), Path(output)
    if output.exists():
        raise ValueError('诊断输出已存在，请保留旧结果并选择新目录')
    rows, manifest, corpus = load_frozen()
    saved = {p['id']: p for p in (json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines())}
    if set(saved) != {r['id'] for r in rows}:
        raise ValueError('保存的向量检索记录与题集不匹配')
    root = Path(__file__).resolve().parents[1]
    report_id = corpus['metainfo']['sha1']
    index = {}
    for i, c in enumerate(corpus['content']['chunks']):
        key = (c['page'], sanitize(c['text']))
        index.setdefault(key, []).append(i)
    replay, lexical, original = [], [], []
    # Fail if future refactors accidentally try to embed/re-rank/generate.
    with patch.object(VectorRetriever, '_get_embedding', side_effect=AssertionError('No API calls in offline replay')):
        retriever = FusionRetriever(root / 'data/stock_data/databases/vector_dbs',
                                    root / 'data/stock_data/databases/chunked_reports', report_id=report_id)
        prepared = retriever.prepare_report(corpus['metainfo']['company_name'])
        for row in rows:
            old = saved[row['id']]
            if old['question'] != row['question']:
                raise ValueError('题目文本与保存记录不匹配')
            candidates = []
            for c in old['retrieved']:
                key = (c['page'], c['text'])
                if key not in index:
                    raise ValueError('旧检索文本不属于冻结语料')
                if len(index[key]) != 1:
                    raise ValueError('脱敏文本不能唯一定位原分块，不猜测身份')
                chunk_index = index[key][0]
                candidates.append(dict(c, text=corpus['content']['chunks'][chunk_index]['text'],
                                       pdf_sha1=report_id, chunk_id=f'original:{chunk_index}', source_pages=[c['page']]))
            with patch.object(retriever.vector, 'retrieve_by_company_name', return_value=candidates):
                result = retriever.retrieve_by_company_name(corpus['metainfo']['company_name'], row['question'], top_n=10)
            replay.append({'id': row['id'], 'retrieved': result})
            lexical.append({'id': row['id'], 'retrieved': retriever.lexical_candidates(corpus['metainfo']['company_name'], row['question'], 10)})
            original.append({'id': row['id'], 'retrieved': old['retrieved']})
    overlay = root / 'data/stock_data/databases/retrieval_overlays' / (report_id + '.json')
    report = dict(mode='offline_saved_vector_replay', external_calls=0, generated_answers=0,
                  caveat='Saved original vector top10 is incomplete for production top30, and repaired text changed. Not a live quality uplift or new answer score.',
                  dataset_sha256=manifest['dataset_sha256'], saved_predictions_sha256=digest(predictions_path),
                  overlay_sha256=digest(overlay), source_integrity='frozen PDF/corpus/vector hashes verified unchanged',
                  code_version=code_version(root), replaced_pages=sorted(prepared['replaced']),
                  old_vector_top10=aggregate_retrieval(rows, original),
                  repaired_bm25_top10=aggregate_retrieval(rows, lexical),
                  fusion_with_saved_vector_top10=aggregate_retrieval(rows, replay),
                  failures=[{'id': row['id'], 'gold_pages': [e['pdf_page'] for e in row['evidence']],
                             'retrieved_pages': [x['page'] for x in result['retrieved']]}
                            for row, result in zip(rows, replay) if row['answerable']
                            and not any(x['page'] in {e['pdf_page'] for e in row['evidence']} for x in result['retrieved'])])
    output.mkdir(parents=True)
    for name, data in [('report.json', report), ('fusion_replay.json', replay), ('bm25_repaired.json', lexical)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('code_version',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.saved_predictions, args.output)
