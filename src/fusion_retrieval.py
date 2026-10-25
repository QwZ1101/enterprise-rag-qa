"""Chinese BM25 + vector retrieval, deterministic reciprocal-rank fusion."""
import re

from rank_bm25 import BM25Okapi

from src.retrieval import VectorRetriever
from src.retrieval_overlay import load_overlay
from trace_logging import emit


def tokenize(text):
    tokens = []
    for run in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', text.lower()):
        if '\u4e00' <= run[0] <= '\u9fff':
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
            if len(run) == 1:
                tokens.append(run)
        else:
            tokens.append(run)
    return tokens


def reciprocal_rank_fusion(vector, lexical, top_n=10, k=60):
    if k <= 0 or top_n < 0:
        raise ValueError('Invalid fusion limits')
    combined = {}
    for source, candidates in [('vector', vector), ('bm25', lexical)]:
        seen = set()
        for rank, candidate in enumerate(candidates, 1):
            key = (candidate['pdf_sha1'], candidate['chunk_id'])
            if key in seen:
                continue
            seen.add(key)
            if key not in combined:
                combined[key] = dict(candidate, rrf_score=0.0, retrieval_ranks={})
            item = combined[key]
            item['rrf_score'] += 1 / (k + rank)
            item['retrieval_ranks'][source] = rank
    result = sorted(combined.values(), key=lambda c: (-c['rrf_score'], c['pdf_sha1'], str(c['chunk_id'])))
    # Raw FAISS distances and BM25 scores have different scales; never add them.
    for candidate in result:
        candidate['distance'] = candidate['rrf_score']
    return result[:top_n]


class FusionRetriever:
    def __init__(self, vector_db_dir, documents_dir, report_id=None, candidate_size=30, overlay_dir=None):
        self.vector = VectorRetriever(vector_db_dir, documents_dir)
        self.report_id = report_id
        self.candidate_size = candidate_size
        self.overlay_dir = overlay_dir
        self._prepared = {}

    def prepare_report(self, company_name):
        report = self.vector.select_report(company_name, self.report_id)
        sha = report['document']['metainfo']['sha1']
        if sha in self._prepared:
            return self._prepared[sha]
        document = report['document']
        overlay = load_overlay(report['document_path'], document, self.overlay_dir)
        replaced = set(overlay['replaced_pages']) if overlay else set()
        chunks = []
        for index, chunk in enumerate(document['content']['chunks']):
            if chunk['page'] not in replaced:
                chunks.append(dict(chunk, pdf_sha1=sha, chunk_id=f'original:{index}',
                                   source_pages=[chunk['page']], provenance='mineru_original'))
        if overlay:
            chunks.extend(dict(c, pdf_sha1=sha, chunk_id=c['id']) for c in overlay['chunks'])
        if not chunks:
            raise ValueError('报告没有可用分块')
        bm25 = BM25Okapi([tokenize(c['text']) or ['__empty__'] for c in chunks], k1=1.5, b=0.75)
        pages = {p['page']: p for p in document['content']['pages'] if p['page'] not in replaced}
        if overlay:
            pages.update({p['page']: p for p in overlay['pages']})
        prepared = dict(report=report, chunks=chunks, bm25=bm25, pages=pages, replaced=replaced,
                        excluded_chunks=sum(c['page'] in replaced for c in document['content']['chunks']))
        self._prepared[sha] = prepared
        return prepared

    def lexical_candidates(self, company_name, query, limit=None):
        data = self.prepare_report(company_name)
        scores = data['bm25'].get_scores(tokenize(query))
        # Zero-score chunks are not evidence; preserve deterministic ties.
        indices = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i))
        return [dict(data['chunks'][i], bm25_score=float(scores[i])) for i in indices[:limit or self.candidate_size]]

    def retrieve_by_company_name(self, company_name, query, top_n=10, return_parent_pages=False, **kwargs):
        if top_n <= 0:
            return []
        data = self.prepare_report(company_name)
        size = max(self.candidate_size, top_n)
        # Fetch extra old-vector candidates to compensate for intentionally excluded pages.
        raw = self.vector.retrieve_by_company_name(company_name, query, top_n=size + data['excluded_chunks'],
                                                    report_id=self.report_id)
        vector = [c for c in raw if c['page'] not in data['replaced']][:size]
        lexical = self.lexical_candidates(company_name, query, size)
        result = reciprocal_rank_fusion(vector, lexical, top_n=size if return_parent_pages else top_n)
        if return_parent_pages:
            unique = []
            seen = set()
            for item in result:
                if item['page'] not in seen:
                    seen.add(item['page'])
                    unique.append(dict(item, text=data['pages'][item['page']]['text']))
            result = unique[:top_n]
        emit('retrieval_policy', mode='bm25_vector_rrf', rrf_k=60, candidate_size=size,
             vector_candidates=len(vector), lexical_candidates=len(lexical),
             repaired_pages=sorted(data['replaced']), reranker_calls=0)
        return result

    def retrieve_all(self, company_name):
        data = self.prepare_report(company_name)
        sha = data['report']['document']['metainfo']['sha1']
        return [dict(p, pdf_sha1=sha, chunk_id=f'page:{p["page"]}', source_pages=[p['page']], distance=0)
                for p in sorted(data['pages'].values(), key=lambda p: p['page'])]
