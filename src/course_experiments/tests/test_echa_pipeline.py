import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from course_experiments.manifest import digest,file_record
from course_experiments.evaluation.echa_pipeline import FrozenTokenizer,candidate_sets,verify_candidates
from course_experiments.tests.test_dual_view_rag import retriever


class Provider:
    def __init__(self,corrupt=False):self.calls=0;self.corrupt=corrupt
    def chat_response(self,messages,**kwargs):
        self.calls+=1;payload=json.loads(messages[-1]['content']);out=[]
        for pair in payload['pairs']:
            out.append(dict(pair_id=pair['pair_id'],decision='accept',role='explanation',same_sense=True,
                scope_consistent=True,evidence_supported=True,reason='synthetic',citations=[dict(
                    evidence_id=pair['candidate']['evidence_id'],quote='fabricated' if self.corrupt else pair['candidate']['text'])]))
        return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(verdicts=out))))])


class ECHAPipelineTests(unittest.TestCase):
    def test_indexed_name_boundaries_equal_frozen_matcher_for_underscore_names(self):
        from course_experiments.evaluation.echa_pipeline import occurrences
        from course_experiments.evaluation.graph_rag_v2 import matches
        entities=[dict(entity_id='left',canonical_name='_reg'),dict(entity_id='right',canonical_name='reg_')]
        structure=dict(evidence={'p':dict(text='x_reg and reg_y')})
        actual=occurrences(entities,structure)
        for entity in entities:
            expected={'p'} if matches(entity['canonical_name'],structure['evidence']['p']['text']) else set()
            self.assertEqual(actual.get(entity['entity_id'],set()),expected)

    def test_indexed_candidates_equal_core_channels_without_full_text_scan_per_concept(self):
        from course_experiments.evaluation.evidence_hierarchical_alignment import candidate_paragraphs
        old=retriever();old.load_book('A')
        actual=list(candidate_sets(old.graph,old.structure,old.mappings,old.ids,old.concept_vectors,old.rows,old.vectors))
        for row in actual:
            cid=row['concept_id'];scores=old.vectors@old.concept_vectors[old.position[cid]]
            expected=candidate_paragraphs(old.entities[cid],old.structure,
                {r['evidence_ids'][0]:float(x) for r,x in zip(old.rows,scores)},origins=old.mappings['concepts'][cid]['A'])
            self.assertEqual(row['candidates'],expected)

    def test_same_description_distinct_concept_ids_have_distinct_pair_identities(self):
        old=retriever();old.load_book('A');graph=copy.deepcopy(old.graph)
        graph['entities'].append(dict(graph['entities'][0],entity_id='cache_duplicate'))
        plans=[dict(concept_id=cid,candidates=[dict(book_id='A',evidence_id='e1',semantic_score=1.,channels=['origin'])]) for cid in ['cache','cache_duplicate']]
        with tempfile.TemporaryDirectory() as d:
            provider=Provider();rows=verify_candidates(provider,graph,old.structure,plans,Path(d),'echa/test')
            self.assertEqual(provider.calls,1);self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='accepted' for r in rows))

    def test_verified_batch_resumes_without_model_call_and_changed_input_rejects(self):
        old=retriever();old.load_book('A');plan=list(candidate_sets(old.graph,old.structure,old.mappings,old.ids,old.concept_vectors,old.rows,old.vectors))
        with tempfile.TemporaryDirectory() as d:
            provider=Provider();first=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            count=provider.calls;again=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            self.assertEqual(first,again);self.assertEqual(provider.calls,count)
            self.assertTrue(all(r['status']=='accepted' for r in first))
            changed=copy.deepcopy(old.structure);changed['evidence']['e1']['text']='changed'
            with self.assertRaises(ValueError):verify_candidates(provider,old.graph,changed,plan,Path(d),'echa/test')

    def test_bad_quotes_get_one_repair_then_unresolved_without_false_support(self):
        old=retriever();old.load_book('A');plan=list(candidate_sets(old.graph,old.structure,old.mappings,old.ids,old.concept_vectors,old.rows,old.vectors))[:1]
        with tempfile.TemporaryDirectory() as d:
            provider=Provider(corrupt=True);rows=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            self.assertEqual(provider.calls,2)
            self.assertTrue(all(r['status']=='rejected_or_unresolved' and r['weight']==0 for r in rows))
            self.assertTrue(list(Path(d).glob('raw/*/*.json')))

    def test_saved_response_before_mapping_commit_is_replayed_without_new_call(self):
        old=retriever();old.load_book('A');plan=list(candidate_sets(old.graph,old.structure,old.mappings,old.ids,old.concept_vectors,old.rows,old.vectors))[:1]
        with tempfile.TemporaryDirectory() as d:
            provider=Provider();first=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            calls=provider.calls
            for p in (Path(d)/'mappings').glob('*.json'):p.unlink()
            again=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            self.assertEqual(again,first);self.assertEqual(provider.calls,calls)

    def test_partial_mapping_commit_retains_original_fixed_batch_identity(self):
        old=retriever();old.load_book('A');plan=list(candidate_sets(old.graph,old.structure,old.mappings,old.ids,old.concept_vectors,old.rows,old.vectors))
        with tempfile.TemporaryDirectory() as d:
            provider=Provider();first=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            calls=provider.calls;files=list((Path(d)/'mappings').glob('*.json'));self.assertGreater(len(files),1)
            files[0].unlink()
            again=verify_candidates(provider,old.graph,old.structure,plan,Path(d),'echa/test')
            self.assertEqual(again,first);self.assertEqual(provider.calls,calls)

    def test_tokenizer_manifest_checks_frozen_identity_and_real_encoder_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tokenizer.json';p.write_text('{}')
            manifest=dict(version='echa_frozen_tokenizer_v1',model='deepseek-ai/DeepSeek-V4-Flash',
                          files={'tokenizer.json':file_record(p)},package_version='test',add_special_tokens=False)
            class Encoder:
                def encode(self,text,add_special_tokens):
                    if add_special_tokens:raise AssertionError()
                    return type('Encoding',(),{'ids':[1,2]})()
            t=FrozenTokenizer(manifest,loader=lambda path:Encoder())
            self.assertEqual(t.count('这不是按字数计数的文本'),2)
            p.write_text('changed')
            with self.assertRaises(ValueError):FrozenTokenizer(manifest,loader=lambda path:Encoder())


if __name__=='__main__':unittest.main()
