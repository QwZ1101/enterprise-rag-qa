"""Offline integration of existing RAG workflow; retriever/model are test doubles."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from src.questions_processing import QuestionsProcessor


class FakeRetriever:
    def __init__(self, **kwargs):
        pass

    def retrieve_by_company_name(self, **kwargs):
        return [{'page': 5, 'text': 'Offline fixture revenue is 123.'}]


class FakeProcessor:
    response_data = {'model': 'offline_fixture', 'input_tokens': 0, 'output_tokens': 0}

    def get_answer_from_rag_context(self, **kwargs):
        return {'final_answer': '123 (offline fixture)', 'relevant_pages': [5]}

    def get_rephrased_questions(self, original_question, companies):
        return {company: company + ' revenue?' for company in companies}


class RAGIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'AGENT_TRACE_DIR': str(self.root), 'AGENT_TRACE_MODE': 'offline_test'})
        self.env.start()
        # Skip external-client initialization; execute real question processing methods.
        self.processor = QuestionsProcessor.__new__(QuestionsProcessor)
        for name, value in {'llm_reranking': False, 'vector_db_dir': 'fixture', 'documents_dir': 'fixture',
                            'full_context': False, 'llm_reranking_sample_size': 5, 'top_n_retrieval': 3,
                            'return_parent_pages': False, 'answering_model': 'offline_fixture',
                            'new_challenge_pipeline': False, 'openai_processor': FakeProcessor()}.items():
            setattr(self.processor, name, value)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def events(self):
        paths = list(self.root.rglob('*.jsonl'))
        self.assertEqual(len(paths), 1)
        return [json.loads(line) for line in paths[0].read_text(encoding='utf-8').splitlines()]

    def test_input_retrieval_model_and_final_output(self):
        with patch('src.questions_processing.VectorRetriever', FakeRetriever):
            result = self.processor.process_single_question('"FixtureCo" revenue?')
        events = self.events()
        calls = [e for e in events if e['event'] == 'call_start']
        self.assertEqual([e['data']['name'] for e in calls], ['retrieval.by_company', 'model.answer_from_context'])
        self.assertEqual(calls[0]['data']['kwargs']['top_n'], 3)
        self.assertIn('Offline fixture', calls[1]['data']['kwargs']['rag_context'])
        self.assertEqual(events[-1]['data']['output'], result)
        self.assertEqual(events[0]['mode'], 'offline_test')

    def test_retrieval_failure_is_recorded(self):
        with patch('src.questions_processing.VectorRetriever', FakeRetriever), patch.object(FakeRetriever, 'retrieve_by_company_name', side_effect=RuntimeError('offline retrieval failure')):
            with self.assertRaisesRegex(RuntimeError, 'offline retrieval failure'):
                self.processor.process_single_question('"FixtureCo" revenue?')
        events = self.events()
        self.assertEqual(events[-1]['data']['status'], 'failed')
        self.assertTrue(any(e['event'] == 'call_failed' for e in events))

    def test_parallel_company_workers_share_request_trace(self):
        with patch('src.questions_processing.VectorRetriever', FakeRetriever):
            self.processor.process_single_question('Compare "FixtureA" and "FixtureB" revenue')
        events = self.events()
        retrievals = [e for e in events if e['event'] == 'call_start' and e['data']['name'] == 'retrieval.by_company']
        self.assertEqual({e['data']['kwargs']['company_name'] for e in retrievals}, {'FixtureA', 'FixtureB'})
        self.assertEqual(len({e['trace_id'] for e in events}), 1)

    def test_recorded_citation_does_not_add_unclaimed_page(self):
        # Actual live trace 8ea6d41504e940509b8a7aedf6ff230e, 2026-10-21:
        # model claimed [7]; old post-processing changed it to [7, 84].
        retrieved = [{'page': p} for p in [7, 84, 5, 2, 14, 195, 6, 1, 195, 8]]
        self.assertEqual(self.processor._validate_page_references([7], retrieved), [7])

    def test_missing_citations_are_not_fabricated(self):
        self.assertEqual(self.processor._validate_page_references(None, [{'page': 7}, {'page': 84}]), [])

    def test_invalid_and_duplicate_citations_are_filtered(self):
        self.assertEqual(self.processor._validate_page_references([7, 7, 999, -1, True, '7'], [{'page': 7}, {'page': 84}]), [7])

    def test_web_bad_request_has_failed_trace(self):
        from server import AskRequest, ask
        response = ask(AskRequest(question=''))
        self.assertEqual(response.status_code, 400)
        end = self.events()[-1]['data']
        self.assertEqual(end['status'], 'failed')
        self.assertEqual(end['output']['status_code'], 400)

    def test_web_caught_pipeline_exception_has_failed_trace(self):
        import server
        with patch.object(server.pipeline, 'answer_single_question', side_effect=RuntimeError('offline pipeline failure')):
            response = server.ask(server.AskRequest(question='中芯国际的总部位于哪个城市？'))
        self.assertEqual(response.status_code, 500)
        events = self.events()
        self.assertTrue(any(e['event'] == 'handled_error' for e in events))
        self.assertEqual(events[-1]['data']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
