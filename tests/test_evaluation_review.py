import unittest
from evaluation.review_report import review_summary


class ReviewTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(id='a',answerable=True),dict(id='b',answerable=False)]
        predictions=[dict(id='a',status='completed',retrieved=[dict(page=7,text='公司缩写 SMIC')]),
                     dict(id='b',status='completed',retrieved=[])]
        review=dict(reviewer='AI assisted',human_reviewed=False,cases=[
            dict(id='a',correctness='correct',support='supported',reason='matches excerpt',
                 supporting_excerpts=[dict(page=7,quote='公司缩写 SMIC')]),
            dict(id='b',correctness='correct',support='abstained',reason='N/A')])
        return rows,predictions,review

    def test_abstentions_excluded_from_support_denominator(self):
        result=review_summary(*self.fixture())
        self.assertEqual(result['correct'],2)
        self.assertEqual(result['non_abstaining_answers'],1)
        self.assertEqual(result['correct_abstentions'],1)

    def test_fabricated_review_evidence_is_rejected(self):
        rows,predictions,review=self.fixture()
        review['cases'][0]['supporting_excerpts'][0]['quote']='invented evidence'
        with self.assertRaises(ValueError): review_summary(rows,predictions,review)

    def test_duplicate_or_missing_reviews_are_rejected(self):
        rows,predictions,review=self.fixture()
        review['cases'][1]=review['cases'][0]
        with self.assertRaises(ValueError): review_summary(rows,predictions,review)

    def test_uncertainty_reported_without_silent_exclusion(self):
        rows,predictions,review=self.fixture()
        review['cases'][0]['support']='uncertain'
        result=review_summary(rows,predictions,review)
        self.assertEqual(result['non_abstaining_answers'],1)
        self.assertEqual(result['support_uncertain'],1)
        self.assertEqual(result['uncertainty_upper_bound'],1)

if __name__=='__main__': unittest.main()
