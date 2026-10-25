"""Deterministic diagnostic metrics. No network, model judge or silent exclusions."""
import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def normalize(value):
    return re.sub(r'[\s,，|]', '', unicodedata.normalize('NFKC', str(value))).lower()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_frozen():
    dataset = HERE/'smic2024_pilot_v1.jsonl'
    meta = json.loads((HERE/'smic2024_pilot_v1.manifest.json').read_text(encoding='utf-8'))
    if digest(dataset) != meta['dataset_sha256']:
        raise ValueError('Frozen dataset hash mismatch')
    for key, name in [('corpus_sha256','corpus_file'), ('vector_sha256','vector_file')]:
        if digest(ROOT/meta[name]) != meta[key]:
            raise ValueError('Frozen corpus/index hash mismatch: '+name)
    if digest(meta['source_pdf']) != meta['pdf_sha256']:
        raise ValueError('Frozen source PDF hash mismatch')
    rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != meta['questions'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Invalid dataset count or duplicate IDs')
    corpus = json.loads((ROOT/meta['corpus_file']).read_text(encoding='utf-8'))
    return rows, meta, corpus

def is_abstention(value):
    # Only the documented structured sentinel counts automatically.
    return isinstance(value, str) and value.strip().upper() == 'N/A'

def strict_match(row, answer):
    if answer is None or isinstance(answer, bool):
        return False
    if not row['answerable']:
        return is_abstention(answer)
    if row['kind'] == 'number':
        try:
            actual = Decimal(normalize(answer))
            expected = Decimal(str(row['gold_answer']))
            return actual.is_finite() and actual == expected
        except (InvalidOperation, ValueError):
            return False
    # Substring containment is deliberately NOT considered correctness.
    return normalize(answer).strip('。.!！') in {
        normalize(x).strip('。.!！') for x in [row['gold_answer']]+row['accepted_aliases']}

def has_markers(text, evidence):
    normalized = normalize(text)
    return all(normalize(evidence[key]) in normalized for key in ('field','literal'))

def retrieval_scores(row, retrieved):
    if not row['answerable']:
        return None
    gold_pages = {e['pdf_page'] for e in row['evidence']}
    first = next((i+1 for i,x in enumerate(retrieved[:10]) if x['page'] in gold_pages), None)
    result = {f'page_hit_at_{k}': any(x['page'] in gold_pages for x in retrieved[:k]) for k in (1,5,10)}
    result['page_mrr_at_10'] = 1/first if first else 0.0
    # A stronger lexical diagnostic, still not semantic relevance or exhaustive recall.
    result['marked_evidence_hit_at_10'] = any(
        x['page']==e['pdf_page'] and has_markers(x['text'], e)
        for x in retrieved[:10] for e in row['evidence'])
    return result

def aggregate_retrieval(rows, predictions):
    eligible = [r for r in rows if r['answerable']]
    by_id = {p['id']:p for p in predictions}
    scores = [retrieval_scores(r, by_id.get(r['id'],{}).get('retrieved',[])) for r in eligible]
    keys = ['page_hit_at_1','page_hit_at_5','page_hit_at_10','page_mrr_at_10','marked_evidence_hit_at_10']
    return {'denominator':len(eligible), **{k:sum(s[k] for s in scores)/len(eligible) for k in keys}}

def summarize(rows, predictions):
    ids = [p['id'] for p in predictions]
    if len(ids)!=len(set(ids)) or set(ids)-{r['id'] for r in rows}:
        raise ValueError('Duplicate or unknown prediction IDs')
    by_id = {p['id']:p for p in predictions}
    scored=[]
    for row in rows:
        p=by_id.get(row['id'],{})
        answer=p.get('output',{}).get('final_answer')
        ok=p.get('status')=='completed'
        scored.append(dict(id=row['id'], answerable=row['answerable'], prediction=answer,
            strict_correct=bool(ok and strict_match(row,answer)),
            abstained=bool(ok and is_abstention(answer)), status=p.get('status','missing'),
            retrieval=retrieval_scores(row,p.get('retrieved',[]))))
    answered=[s for s in scored if s['answerable']]
    negatives=[s for s in scored if not s['answerable']]
    return dict(total=len(rows), completed=sum(s['status']=='completed' for s in scored),
        strict_correct=sum(s['strict_correct'] for s in scored),
        strict_accuracy=sum(s['strict_correct'] for s in scored)/len(rows),
        answerable_correct=sum(s['strict_correct'] for s in answered), answerable_total=len(answered),
        correct_abstentions=sum(s['strict_correct'] for s in negatives), unanswerable_total=len(negatives),
        retrieval=aggregate_retrieval(rows,predictions), cases=scored)
