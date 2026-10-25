"""Offline BM25/source audit, or explicitly authorized bounded live evaluation."""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.metrics import load_frozen, digest, has_markers, summarize, aggregate_retrieval

def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def tokenize(text):
    # Fixed tokenizer: Han bigrams within runs plus ASCII words/numbers; no fitted gold features.
    tokens=[]
    for run in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+',text.lower()):
        if '\u4e00' <= run[0] <= '\u9fff':
            tokens.extend(run[i:i+2] for i in range(len(run)-1))
            if len(run)==1: tokens.append(run)
        else: tokens.append(run)
    return tokens

def offline(out):
    from rank_bm25 import BM25Okapi
    rows,meta,corpus=load_frozen()
    chunks=corpus['content']['chunks']
    bm25=BM25Okapi([tokenize(c['text']) for c in chunks], k1=1.5, b=0.75)
    results=[]
    for row in rows:
        scores=bm25.get_scores(tokenize(row['question']))
        indices=sorted(range(len(chunks)),key=lambda i:(-scores[i],i))[:10]
        results.append({'id':row['id'],'retrieved':[dict(page=chunks[i]['page'],text=chunks[i]['text'],
            chunk_id=chunks[i]['id'],score=float(scores[i])) for i in indices]})
    pages={p['page']:p['text'] for p in corpus['content']['pages']}
    fields=[dict(id=r['id'], page=r['evidence'][0]['pdf_page'],
        page_present=r['evidence'][0]['pdf_page'] in pages,
        markers_preserved=any(has_markers(pages.get(e['pdf_page'],''),e) for e in r['evidence']))
        for r in rows if r['answerable']]
    save(out/'bm25_predictions.json',results)
    save(out/'offline_metrics.json',dict(dataset_sha256=meta['dataset_sha256'], mode='offline',
        external_calls=0, baseline='NEW BM25Okapi k1=1.5 b=0.75; Han bigram + ASCII; same 738 chunks',
        retrieval=aggregate_retrieval(rows,results), field_checks=fields,
        field_marker_preserved=sum(x['markers_preserved'] for x in fields), field_marker_total=len(fields),
        missing_pdf_pages=sorted(set(range(1,meta['source_pages']+1))-set(pages)),
        interpretation='字段和值在指定页共现率，不是完整PDF解析准确率；指定证据命中不是全面召回率。'))
    print('Offline audit and BM25 baseline saved: '+str(out),flush=True)

def live(out):
    rows,meta,_=load_frozen()
    # Never resume automatically or overwrite a partial/failed run.
    result_path=out/'predictions.jsonl'
    if result_path.exists() or (out/'run_manifest.json').exists():
        raise ValueError('Run directory already used; retain it. New runs require new authorization.')
    os.environ['AGENT_TRACE_MODE']='live'
    os.environ['AGENT_TRACE_DIR']=str(out/'traces')
    from src.questions_processing import QuestionsProcessor
    from src.retrieval import VectorRetriever
    from trace_logging import Trace, sanitize, record_error, emit, code_version
    from trace_export import export_trace
    import dashscope
    import dashscope.api_entities.http_request as http_module
    # Ensure only the approved domain receives content, regardless of inherited SDK config.
    dashscope.base_http_api_url='https://dashscope.aliyuncs.com/api/v1'
    processor=QuestionsProcessor(vector_db_dir=ROOT/'data/stock_data/databases/vector_dbs',
        documents_dir=ROOT/'data/stock_data/databases/chunked_reports',
        subset_path=ROOT/'data/stock_data/subset.csv',new_challenge_pipeline=True,
        parent_document_retrieval=False,llm_reranking=False,top_n_retrieval=10,
        parallel_requests=1,api_provider='dashscope',answering_model='qwen3.8-flash')
    base=processor.openai_processor.processor
    base.client=base.client.with_options(max_retries=0,timeout=60.0)
    original_create=base.client.chat.completions.create
    original_embedding=dashscope.TextEmbedding.call
    original_retrieve=VectorRetriever.retrieve_by_company_name
    state={}
    def create(*args,**kwargs):
        if state['generation_calls']>=2: raise RuntimeError('Per-question generation budget exhausted')
        state['generation_calls']+=1
        kwargs['max_tokens']=2048
        response=original_create(*args,**kwargs)
        usage=response.usage.model_dump() if response.usage else {}
        state['generation_usage'].append(usage)
        emit('eval_api_usage',provider='dashscope',kind='generation',usage=usage,
             model=response.model,request_id=response.id,call_number=state['generation_calls'])
        return response
    def embedding(*args,**kwargs):
        if state['embedding_calls']>=1: raise RuntimeError('Per-question embedding budget exhausted')
        state['embedding_calls']+=1
        kwargs['timeout']=30
        result=original_embedding(*args,**kwargs)
        state['embedding_usage']=dict(result.get('usage') or {})
        emit('eval_api_usage',provider='dashscope',kind='embedding',usage=state['embedding_usage'],
             request_id=result.get('request_id'),status_code=result.get('status_code'))
        return result
    def retrieve(self,*args,**kwargs):
        result=original_retrieve(self,*args,**kwargs)
        state['retrieved']=result
        return result
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),historical_evidence=False,
        dataset_sha256=meta['dataset_sha256'],corpus_sha256=meta['corpus_sha256'],vector_sha256=meta['vector_sha256'],
        source_version=code_version(ROOT), model='qwen3.8-flash',embedding_model='qwen3.7-text-embedding-flash',
        top_k=10,reranking=False,parent_pages=False,temperature=0.1,
        harness_overrides='generation max_tokens=2048 timeout=60 retry=0; embedding timeout=30 retry=0',
        max_questions=24,max_embedding_calls=24,max_generation_calls=48,
        annotation='AI辅助标注，待人工复核；一次新诊断评测，无历史基线',
        harness_sha256=digest(__file__),metrics_sha256=digest(ROOT/'evaluation/metrics.py'))
    save(out/'run_manifest.json',manifest)
    predictions=[]
    with patch.object(base.client.chat.completions,'create',side_effect=create), \
         patch.object(dashscope.TextEmbedding,'call',side_effect=embedding), \
         patch.object(http_module,'_send_with_retry',side_effect=lambda send:send()), \
         patch.object(VectorRetriever,'retrieve_by_company_name',new=retrieve):
        for row in rows:
            state.clear()
            state.update(embedding_calls=0,generation_calls=0,generation_usage=[],retrieved=[])
            trace=Trace('rag',ROOT,{'question':row['question'],'kind':row['kind']},
                {'evaluation_id':row['id'],'dataset_sha256':meta['dataset_sha256'],'historical_record':False})
            started=time.monotonic()
            record=dict(id=row['id'],question=row['question'],status='failed')
            with trace.activate(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    output=processor.process_single_question(row['question'],kind=row['kind'])
                    record.update(status='completed',output=output)
                    trace.finish('completed',output)
                except Exception as error:
                    record_error(error)
                    trace.finish('failed')
                    record.update(error_type=type(error).__name__,error=str(error))
            record.update(state,elapsed_seconds=round(time.monotonic()-started,3),trace_id=trace.id)
            exported=export_trace(trace.path,out/(trace.id+'_export.json'))
            record['trace_complete']=exported['complete_record']
            safe=sanitize(record)
            with result_path.open('a',encoding='utf-8') as fp:
                fp.write(json.dumps(safe,ensure_ascii=False)+'\n')
            predictions.append(safe)
            print(json.dumps(dict(id=row['id'],status=safe['status'],elapsed_seconds=safe['elapsed_seconds'],
                embedding_calls=state['embedding_calls'],generation_calls=state['generation_calls']),ensure_ascii=False),flush=True)
    save(out/'automatic_metrics.json',summarize(rows,predictions))
    print('Completed all 24 cases. Failed cases retained; no automatic reruns.',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['offline','live'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-external-24',action='store_true',help='Run only after explicit authorization for this batch.')
    args=parser.parse_args()
    if args.mode=='live' and not args.allow_external_24:
        parser.error('Live mode requires explicit authorization and --allow-external-24')
    args.out.mkdir(parents=True,exist_ok=True)
    (offline if args.mode=='offline' else live)(args.out.resolve())

if __name__=='__main__':
    main()
