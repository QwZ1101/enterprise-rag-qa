"""Validate an explicit per-case review and aggregate it without changing predictions."""
import argparse
import json
from pathlib import Path
from evaluation.metrics import digest, load_frozen, summarize, normalize

def review_summary(rows, predictions, review):
    expected={r['id'] for r in rows}
    items=review['cases']
    if len(items)!=len(expected) or {r['id'] for r in items}!=expected:
        raise ValueError('Every case must be reviewed exactly once')
    by_id={p['id']:p for p in predictions}
    if set(by_id)!=expected or len(predictions)!=len(expected):
        raise ValueError('Predictions must include every case exactly once')
    for item in items:
        p=by_id[item['id']]
        if item['correctness'] not in ('correct','incorrect','uncertain','failed'):
            raise ValueError('Invalid correctness label')
        if item['support'] not in ('supported','unsupported','uncertain','abstained','failed'):
            raise ValueError('Invalid support label')
        if not item.get('reason'):
            raise ValueError('Review must state evidence/reason')
        if p['status']!='completed' and (item['correctness']!='failed' or item['support']!='failed'):
            raise ValueError('Execution failures cannot count as answers')
        if item['support']=='supported' and not item.get('supporting_excerpts'):
            raise ValueError('Supported review must identify retrieved evidence')
        for e in item.get('supporting_excerpts',[]):
            if not e['quote'] or not any(c['page']==e['page'] and normalize(e['quote']) in normalize(c['text'])
                for c in p.get('retrieved',[])):
                raise ValueError('Review excerpt not found in actual retrieval: '+item['id'])
    judged={r['id']:r for r in items}
    completed_answers=[r for r in items if r['support'] not in ('abstained','failed')]
    unsupported=sum(r['support']=='unsupported' for r in completed_answers)
    uncertain=sum(r['support']=='uncertain' for r in completed_answers)
    n=len(completed_answers)
    return dict(reviewer=review['reviewer'],human_reviewed=review['human_reviewed'],
        total=len(rows),correct=sum(r['correctness']=='correct' for r in items),
        correct_answerable=sum(judged[r['id']]['correctness']=='correct' for r in rows if r['answerable']),
        answerable_total=sum(r['answerable'] for r in rows),
        correct_abstentions=sum(judged[r['id']]['correctness']=='correct' and judged[r['id']]['support']=='abstained'
            for r in rows if not r['answerable']),
        unanswerable_total=sum(not r['answerable'] for r in rows),
        correctness_uncertain=sum(r['correctness']=='uncertain' for r in items),
        failed=sum(r['correctness']=='failed' for r in items),
        non_abstaining_answers=n,unsupported=unsupported,support_uncertain=uncertain,
        unsupported_rate=unsupported/n if n else None,
        uncertainty_upper_bound=(unsupported+uncertain)/n if n else None)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    run=args.run_dir
    rows,meta,_=load_frozen()
    review=json.loads((run/'case_review.json').read_text(encoding='utf-8'))
    if review['dataset_sha256']!=meta['dataset_sha256'] or review['predictions_sha256']!=digest(run/'predictions.jsonl'):
        raise ValueError('Review refers to different labels or predictions')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
    result={'automatic':summarize(rows,predictions),'review':review_summary(rows,predictions,review)}
    (run/'reviewed_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['review'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
