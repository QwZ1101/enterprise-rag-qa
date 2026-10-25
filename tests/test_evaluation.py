import unittest
from evaluation.metrics import strict_match, retrieval_scores, summarize, is_abstention, load_frozen
from evaluation.run_pilot import tokenize


class EvaluationTests(unittest.TestCase):
    def row(self, ident='Q1', **kwargs):
        value=dict(id=ident,answerable=True,kind='number',gold_answer=1234000,accepted_aliases=[],
                   evidence=[dict(pdf_page=8,field='营业收入',literal='1,234')])
        value.update(kwargs)
        return value

    def test_numeric_units_years_and_invalid_values(self):
        r=self.row()
        self.assertTrue(strict_match(r,'1,234,000'))
        for bad in [1234,1234001,True,None,'NaN','Infinity','1234000元']:
            self.assertFalse(strict_match(r,bad),bad)

    def test_negative_sentence_is_not_substring_correctness(self):
        r=self.row(kind='string',gold_answer='复旦大学')
        self.assertTrue(strict_match(r,'复旦大学。'))
        self.assertFalse(strict_match(r,'不是复旦大学，而是其他大学'))

    def test_only_explicit_sentinel_is_automatic_abstention(self):
        self.assertTrue(is_abstention(' N/A '))
        self.assertFalse(is_abstention('N/A，但可能是123'))
        self.assertFalse(is_abstention(None))

    def test_wrong_page_and_missing_field_do_not_count_as_evidence(self):
        r=self.row()
        x=retrieval_scores(r,[{'page':9,'text':'营业收入 1,234'},{'page':8,'text':'其他收入 1,234'}])
        self.assertFalse(x['page_hit_at_1'])
        self.assertTrue(x['page_hit_at_5'])
        self.assertEqual(x['page_mrr_at_10'],0.5)
        self.assertFalse(x['marked_evidence_hit_at_10'])

    def test_top_k_counts_chunks_not_unique_pages(self):
        chunks=[dict(page=1,text='irrelevant')]*10+[dict(page=8,text='营业收入 1,234')]
        self.assertFalse(retrieval_scores(self.row(),chunks)['page_hit_at_10'])

    def test_errors_and_missing_cases_stay_in_denominator(self):
        rows=[self.row('a'),self.row('b'),self.row('c',answerable=False,gold_answer='N/A')]
        result=summarize(rows,[dict(id='a',status='completed',output={'final_answer':1234000}),
            dict(id='c',status='failed',output={'final_answer':'N/A'})])
        self.assertEqual(result['strict_accuracy'],1/3)
        self.assertEqual(result['correct_abstentions'],0)
        self.assertEqual(result['retrieval']['denominator'],2)

    def test_duplicate_unknown_predictions_rejected(self):
        with self.assertRaises(ValueError):
            summarize([self.row()],[{'id':'Q1'},{'id':'Q1'}])
        with self.assertRaises(ValueError):
            summarize([self.row()],[{'id':'unknown'}])

    def test_chinese_tokenizer(self):
        self.assertEqual(tokenize('营业收入 2024 SMIC'),['营业','业收','收入','2024','smic'])

    def test_frozen_dataset_and_sources(self):
        rows,meta,corpus=load_frozen()
        self.assertEqual(len(rows),24)
        self.assertEqual(sum(r['answerable'] for r in rows),20)
        self.assertEqual(len(corpus['content']['chunks']),738)
        self.assertFalse(meta['human_reviewed'])

if __name__=='__main__':
    unittest.main()
