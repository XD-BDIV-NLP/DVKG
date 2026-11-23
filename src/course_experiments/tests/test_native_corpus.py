import unittest
from course_experiments.evaluation.native_corpus import native_passages


class NativeCorpusTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(book_id='b',chunk_id='c',evidence_ids=[i],text=text)
              for i,text in [('e2','第二段。'),('e1','第一段。')]]
        structure=dict(book_id='b',evidence={
            'e1':dict(text='第一段。',chunk_id='c',chapter_id='chapter',source=dict(paragraph_order=1)),
            'e2':dict(text='第二段。',chunk_id='c',chapter_id='chapter',source=dict(paragraph_order=2))})
        return rows,structure

    def test_original_source_passages_preserve_every_character_and_identifier(self):
        rows,structure=self.fixture()
        docs,audit=native_passages(rows,structure,'b')
        self.assertEqual(len(docs),1)
        self.assertEqual(docs[0]['text'],'第一段。\n第二段。')
        self.assertEqual(docs[0]['evidence_ids'],['e1','e2'])
        for span in docs[0]['evidence_spans']:
            self.assertEqual(docs[0]['text'][span['start']:span['end']],structure['evidence'][span['evidence_id']]['text'])
        self.assertTrue(audit['all_evidence_covered'])
        self.assertEqual(docs,native_passages(list(reversed(rows)),structure,'b')[0])

    def test_wrong_text_or_foreign_book_rejected(self):
        rows,structure=self.fixture();rows[0]['text']='改动'
        with self.assertRaises(ValueError):native_passages(rows,structure,'b')
        rows,structure=self.fixture();rows[0]['book_id']='foreign'
        with self.assertRaises(ValueError):native_passages(rows,structure,'b')

    def test_ambiguous_or_duplicate_evidence_is_rejected(self):
        rows,structure=self.fixture()
        with self.assertRaises(ValueError):native_passages(rows+[rows[0]],structure,'b')
        rows[0]['evidence_ids']=['e1','e2']
        with self.assertRaises(ValueError):native_passages(rows,structure,'b')


if __name__=='__main__':unittest.main()
