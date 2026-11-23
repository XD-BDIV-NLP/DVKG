from __future__ import annotations

import unittest

from course_experiments.evaluation import reference_v2 as ref


def corpus(book):
    return dict(fingerprint='source_'+book,chunks=[dict(id=book+str(i),book_id=book,
        chapter_id='chapter_'+str(i%3),evidence_ids=[book+'e'+str(i)]) for i in range(8)],
        evidence=[dict(id=book+'e'+str(i),text='Alpha consists of Beta.') for i in range(8)])


class ReferenceV2Tests(unittest.TestCase):
    def test_primary_AB_and_CD_supplement_are_separate_source_only_metadata(self):
        books=[dict(alias=a,book_id=a) for a in 'ABCD']
        corpora={a:corpus(a) for a in 'ABCD'}
        plan=ref.prepare_plan('computer_organization',books,corpora,dict(relation_types={'part_of':''}))
        self.assertEqual(plan['target_facts'],400)
        self.assertEqual(plan['primary_book_targets'],dict(A=200,B=200))
        self.assertEqual(plan['supplement_book_targets'],dict(C=100,D=100))
        self.assertEqual(plan['status'],'source_only_prepared_no_model_calls_no_reference_claim')
        self.assertNotIn('Alpha consists of Beta.',str(plan))
        self.assertEqual(len(plan['units']),32)

    def test_order_ignores_input_order_and_payload_rebinds_exact_source(self):
        books=[dict(alias=a,book_id=a) for a in 'AB'];corpora={a:corpus(a) for a in 'AB'}
        schema=dict(relation_types={'part_of':''})
        plan=ref.prepare_plan('os',books,corpora,schema)
        reordered={b:dict(c,chunks=list(reversed(c['chunks']))) for b,c in corpora.items()}
        other=ref.prepare_plan('os',list(reversed(books)),reordered,schema)
        self.assertEqual(plan,other)
        payload=ref.candidate_payload(plan,plan['units'][0]['unit_id'],corpora,schema)
        self.assertEqual(payload['output_limits']['questions'],0)
        self.assertIn('Alpha consists of Beta.',payload['unit']['text'])
        corpora['A']['evidence'][0]['text']='changed'
        changed=next(u for u in plan['units'] if u['chunk_id']=='A0')
        with self.assertRaises(ValueError):ref.candidate_payload(plan,changed['unit_id'],corpora,schema)

    def test_acceptance_reuses_original_exact_quote_validation(self):
        unit=dict(book_id='A',chunk_id='a',text='Alpha consists of Beta.',evidence=[dict(id='e',text='Alpha consists of Beta.')])
        packet=dict(entities=[dict(local_id='a',canonical_name='Alpha',aliases=[],scope='technical',citations=[dict(evidence_id='e',quote='Alpha')]),
            dict(local_id='b',canonical_name='Beta',aliases=[],scope='technical',citations=[dict(evidence_id='e',quote='Beta')])],
            facts=[dict(local_id='f',head='b',tail='a',relation='part_of',conditions='',citations=[dict(evidence_id='e',quote='Alpha consists of Beta.')])],questions=[])
        selected,audit=ref.curate_candidate(packet,unit,dict(relation_types={'part_of':''}))
        self.assertEqual(len(selected['facts']),1)
        self.assertFalse(audit['human_audited'])
        self.assertEqual(audit['semantic_truth_status'],'pending_independent_machine_judgment')
        packet['facts'][0]['citations'][0]['quote']='invented'
        selected,_=ref.curate_candidate(packet,unit,dict(relation_types={'part_of':''}))
        self.assertEqual(selected['facts'],[])


if __name__=='__main__':unittest.main()
