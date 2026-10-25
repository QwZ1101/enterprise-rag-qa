import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from src.fusion_retrieval import FusionRetriever, reciprocal_rank_fusion, tokenize
from src.retrieval import VectorRetriever
from src.retrieval_overlay import digest, load_overlay
from scripts.build_retrieval_overlay import continuation_groups


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / 'databases/chunked_reports'
        self.vectors = self.root / 'databases/vector_dbs'
        self.overlays = self.root / 'databases/retrieval_overlays'
        for p in [self.docs, self.vectors, self.overlays, self.root / 'pdf_reports']:
            p.mkdir(parents=True)
        self.corpus_path = self.docs / 'report.json'
        self.document = {'metainfo': {'sha1': 'report1', 'company_name': 'FixtureCo'}, 'content': {
            'pages': [{'page': 1, 'text': '公司总部位于上海。'}, {'page': 2, 'text': '流动负债合计 123 错位旧页'},
                      {'page': 4, 'text': '研发人员数量百人。'}, {'page': 5, 'text': '供应商采购业务。'}],
            'chunks': [{'id': 0, 'page': 1, 'text': '公司总部位于上海。'},
                       {'id': 1, 'page': 2, 'text': '流动负债合计 123 错位旧页'},
                       {'id': 2, 'page': 4, 'text': '研发人员数量百人。'},
                       {'id': 3, 'page': 5, 'text': '供应商采购业务。'}]}}
        self.corpus_path.write_text(json.dumps(self.document), encoding='utf-8')
        index = faiss.IndexFlatL2(2)
        index.add(np.array([[10, 10], [0, 0], [5, 5], [8, 8]], dtype=np.float32))
        (self.vectors / 'report1.faiss').write_bytes(faiss.serialize_index(index).tobytes())
        pdf = self.root / 'pdf_reports/report.pdf'
        pdf.write_bytes(b'offline source PDF identity fixture')
        self.overlay = dict(version=1, report_id='report1', corpus_sha256=digest(self.corpus_path),
                            pdf_sha256=digest(pdf), pdf_file_name=pdf.name, pdf_page_count=5,
                            replaced_pages=[2, 3],
                            pages=[{'page': 2, 'text': '单位：千元。'}, {'page': 3, 'text': '流动负债合计 123'}],
                            chunks=[{'id': 'physical:2:0', 'page': 2, 'text': '单位：千元。'},
                                    {'id': 'physical:3:0', 'page': 3, 'text': '流动负债合计 123',
                                     'context_pages': [{'page': 2, 'text': '单位：千元。'}]}])
        self.overlay_path = self.overlays / 'report1.json'
        self.overlay_path.write_text(json.dumps(self.overlay), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def retriever(self):
        with patch.object(VectorRetriever, '_set_up_llm', return_value=None):
            return FusionRetriever(self.vectors, self.docs, candidate_size=3)

    def test_chinese_and_numeric_tokenization(self):
        self.assertEqual(tokenize('流动负债 SMIC 2024'), ['流动', '动负', '负债', 'smic', '2024'])

    def test_rank_fusion_deduplicates_by_report_and_chunk_not_page(self):
        a = dict(pdf_sha1='a', chunk_id='1', page=1, text='one', distance=999)
        b = dict(pdf_sha1='a', chunk_id='2', page=1, text='two', distance=-999)
        c = dict(pdf_sha1='b', chunk_id='1', page=1, text='other document')
        result = reciprocal_rank_fusion([a, b], [b, b, c])
        self.assertEqual(result[0]['chunk_id'], '2')
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['retrieval_ranks'], {'vector': 2, 'bm25': 1})

    def test_hybrid_excludes_wrong_page_and_keeps_new_physical_page(self):
        retriever = self.retriever()
        with patch.object(retriever.vector, '_get_embedding', return_value=[0, 0]) as embed:
            result = retriever.retrieve_by_company_name('FixtureCo', '流动负债合计', top_n=4)
        embed.assert_called_once()
        self.assertFalse(any(x['chunk_id'] == 'original:1' for x in result))
        fixed = next(x for x in result if x['chunk_id'] == 'physical:3:0')
        self.assertEqual(fixed['page'], 3)
        self.assertEqual(fixed['context_pages'][0]['page'], 2)
        self.assertEqual(fixed['pdf_sha1'], 'report1')

    def test_bm25_without_vector_or_model_requests(self):
        retriever = self.retriever()
        with patch.object(retriever.vector, '_get_embedding', side_effect=AssertionError('no external call')):
            result = retriever.lexical_candidates('FixtureCo', '流动负债')
            self.assertEqual(result[0]['page'], 3)
            self.assertEqual(retriever.lexical_candidates('FixtureCo', '完全没有匹配xyz987'), [])

    def test_parent_and_full_context_use_repaired_pages(self):
        retriever = self.retriever()
        with patch.object(retriever.vector, '_get_embedding', return_value=[0, 0]):
            result = retriever.retrieve_by_company_name('FixtureCo', '流动负债', return_parent_pages=True)
        self.assertTrue(any(x['page'] == 3 for x in result))
        all_pages = retriever.retrieve_all('FixtureCo')
        self.assertEqual(next(x for x in all_pages if x['page'] == 2)['text'], '单位：千元。')
        self.assertFalse(any('错位旧页' in x['text'] for x in all_pages))

    def test_stale_corpus_or_pdf_overlay_fails_closed(self):
        self.corpus_path.write_text(json.dumps(self.document) + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '语料'):
            self.retriever().prepare_report('FixtureCo')
        self.corpus_path.write_text(json.dumps(self.document), encoding='utf-8')
        (self.root / 'pdf_reports/report.pdf').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'PDF'):
            self.retriever().prepare_report('FixtureCo')

    def test_missing_repair_page_and_invented_context_are_rejected(self):
        self.overlay['chunks'][1]['context_pages'][0]['text'] = 'fabricated units'
        self.overlay_path.write_text(json.dumps(self.overlay), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_overlay(self.corpus_path, self.document)

    def test_ambiguous_company_requires_report_id_before_embedding(self):
        retriever = self.retriever()
        extra = dict(retriever.vector.all_dbs[0], name='report2',
                     document={'metainfo': {'company_name': 'FixtureCo', 'sha1': 'report2'}})
        retriever.vector.all_dbs.append(extra)
        with self.assertRaisesRegex(ValueError, '唯一报告'):
            retriever.vector.select_report('FixtureCo')
        self.assertEqual(retriever.vector.select_report('FixtureCo', 'report1')['name'], 'report1')

    def test_page_context_and_references_keep_pdf_identity(self):
        from src.questions_processing import QuestionsProcessor
        p = QuestionsProcessor.__new__(QuestionsProcessor)
        result = self.retriever().lexical_candidates('FixtureCo', '流动负债')
        context = p._format_retrieval_results(result)
        self.assertIn('page 3', context)
        self.assertIn('page 2', context)
        self.assertEqual(p._validate_page_references([3, 2, 99], result), [3, 2])
        self.assertEqual(p._extract_references([3], 'FixtureCo', result), [{'pdf_sha1': 'report1', 'page_index': 3}])

    def test_continuation_detection_follows_actual_table_sequence(self):
        elements = [dict(type='table', page_idx=0, table_body='body'),
                    dict(type='header', page_idx=1), dict(type='table', page_idx=1),
                    dict(type='table', page_idx=2), dict(type='table', page_idx=7)]
        self.assertEqual(continuation_groups(elements), [[1, 2, 3]])

    def test_web_passes_unique_report_to_pipeline(self):
        import server
        with patch.dict(os.environ, {'AGENT_TRACE_MODE': 'offline_test', 'AGENT_TRACE_DIR': str(self.root / 'traces')}), \
                patch.object(server.pipeline, 'answer_single_question', return_value={'final_answer': 'fixture'}) as answer:
            result = server.ask(server.AskRequest(question='中芯国际总部', report_id='stock_10001'))
        self.assertEqual(result['answer'], 'fixture')
        self.assertEqual(answer.call_args.kwargs['report_id'], 'stock_10001')

    def test_real_pipeline_defaults_to_fusion_and_cites_repaired_evidence(self):
        from src.pipeline import Pipeline
        self.root.joinpath('subset.csv').write_text('sha1,file_name,company_name\nreport1,report,FixtureCo\n', encoding='utf-8')

        class ModelStub:
            response_data = {'model': 'offline_stub', 'input_tokens': 0, 'output_tokens': 0}

            def get_answer_from_rag_context(self, **kwargs):
                context = kwargs['rag_context']
                self_test.assertIn('page 3', context)
                self_test.assertIn('流动负债合计 123', context)
                self_test.assertIn('单位：千元', context)
                self_test.assertNotIn('错位旧页', context)
                return {'final_answer': 123000, 'relevant_pages': [3, 2, 99]}

        self_test = self
        with patch.dict(os.environ, {'AGENT_TRACE_MODE': 'offline_test', 'AGENT_TRACE_DIR': str(self.root / 'traces')}), \
                patch('src.questions_processing.APIProcessor', return_value=ModelStub()), \
                patch.object(VectorRetriever, '_set_up_llm', return_value=None), \
                patch.object(VectorRetriever, '_get_embedding', return_value=[0, 0]):
            result = Pipeline(self.root).answer_single_question('FixtureCo 流动负债合计', kind='number', report_id='report1')
        self.assertEqual(result['relevant_pages'], [3, 2])
        self.assertEqual(result['references'], [{'pdf_sha1': 'report1', 'page_index': 3},
                                                 {'pdf_sha1': 'report1', 'page_index': 2}])

    def test_vector_count_mismatch_stops_before_embedding(self):
        retriever = self.retriever()
        retriever.vector.all_dbs[0]['vector_db'].reset()
        with patch.object(retriever.vector, '_get_embedding', side_effect=AssertionError('No request')):
            with self.assertRaisesRegex(ValueError, '向量条数'):
                retriever.vector.retrieve_by_company_name('FixtureCo', 'question')


if __name__ == '__main__':
    unittest.main()
