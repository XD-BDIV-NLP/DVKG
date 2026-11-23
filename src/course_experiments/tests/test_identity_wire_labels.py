import copy
import unittest

from course_experiments.evaluation import scope_identity as source
from course_experiments.evaluation import scope_identity_batch as core
from course_experiments.evaluation.scope_identity_packing import messages_for
from course_experiments.tests.test_scope_identity_batch import packet


class IdentityWireLabelTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_wire_labels
        return identity_wire_labels

    def answer(self,short):
        def one(pair):
            return dict(candidate_id=pair['candidate_id'],decision='same_sense',reason='synthetic',
                scope_consistent=True,source_definition_sufficient=True,homonym_excluded=True,
                left_citations=[{k:v for k,v in pair['left'].get('evidence_quotes',pair['left'].get('source_quotes'))[0].items() if k!='provenance'}],
                right_citations=[{k:v for k,v in pair['right'].get('evidence_quotes',pair['right'].get('source_quotes'))[0].items() if k!='provenance'}])
        return dict(batch_id=short['batch_id'],results=[one(p) for p in short['pairs']]) if 'pairs' in short else one(short)

    def test_single_bijection_changes_only_structural_ids_not_text(self):
        api=self.api();p=packet(3);wire=source.anonymous_packet(p);before=copy.deepcopy(wire)
        short,table=api.encode_wire(wire)
        self.assertEqual(short['candidate_id'],'P0')
        self.assertEqual(short['left']['source_quotes'][0]['evidence_id'],'L0')
        self.assertEqual(short['right']['source_quotes'][0]['evidence_id'],'R0')
        self.assertEqual(api.restore_wire(short,table),wire)
        self.assertEqual(wire,before)
        for side in ['left','right']:
            self.assertEqual(short[side]['scope'],wire[side]['scope'])
            self.assertEqual(short[side]['model_claimed_definitions'],wire[side]['model_claimed_definitions'])
            self.assertEqual(short[side]['source_quotes'][0]['quote'],wire[side]['source_quotes'][0]['quote'])
        decoded=api.decode_response(self.answer(short),table)
        self.assertTrue(source.validate_judgment(dict(wire,both_sides_have_source=True),decoded)['eligible_identity_edge'])

    def test_batch_and_reverse_restore_original_strict_validator(self):
        api=self.api();import json
        work=[packet(3),packet(4)]
        for reverse in [False,True]:
            wire=json.loads(messages_for(work,reverse=reverse)[1]['content']);short,table=api.encode_wire(wire)
            self.assertEqual(short['batch_id'],'B0')
            self.assertEqual(api.restore_wire(short,table),wire)
            decoded=api.decode_response(self.answer(short),table)
            for p,row in zip(work,decoded['results']):
                self.assertTrue(core._validate(p,core._anonymous(p),row,reverse)['eligible_identity_edge'])

    def test_unknown_cross_side_cross_pair_labels_rejected_without_dropping_quotes(self):
        api=self.api();import json
        wire=json.loads(messages_for([packet(3),packet(4)])[1]['content']);short,table=api.encode_wire(wire)
        for label in ['unknown',short['pairs'][0]['left']['evidence_quotes'][0]['evidence_id'],short['pairs'][1]['right']['evidence_quotes'][0]['evidence_id']]:
            result=self.answer(short);result['results'][0]['right_citations'][0]['evidence_id']=label;before=copy.deepcopy(result)
            with self.assertRaises(ValueError):api.decode_response(result,table)
            self.assertEqual(result,before)

    def test_full_hash_or_prefix_cannot_be_guessed_as_label(self):
        api=self.api();wire=source.anonymous_packet(packet(3));short,table=api.encode_wire(wire)
        for label in [wire['candidate_id'],wire['candidate_id'][:8],'P00','p0']:
            result=self.answer(short);result['candidate_id']=label
            with self.assertRaises(ValueError):api.decode_response(result,table)

    def test_restoring_ids_does_not_make_fabricated_quote_valid(self):
        api=self.api();wire=source.anonymous_packet(packet(3));short,table=api.encode_wire(wire)
        result=self.answer(short);result['right_citations'][0]['quote']='fabricated source'
        decoded=api.decode_response(result,table)
        with self.assertRaises(ValueError):source.validate_judgment(dict(wire,both_sides_have_source=True),decoded)

    def test_missing_duplicate_batch_candidates_and_changed_batch_id_rejected(self):
        api=self.api();import json
        wire=json.loads(messages_for([packet(3),packet(4)])[1]['content']);short,table=api.encode_wire(wire)
        bad=self.answer(short);bad['results'].pop()
        with self.assertRaises(ValueError):api.decode_response(bad,table)
        bad=self.answer(short);bad['results'][1]=copy.deepcopy(bad['results'][0])
        with self.assertRaises(ValueError):api.decode_response(bad,table)
        bad=self.answer(short);bad['batch_id']='B1'
        with self.assertRaises(ValueError):api.decode_response(bad,table)

    def test_prior_provenance_is_preserved_and_original_prior_validator_still_applies(self):
        api=self.api();import json
        p=packet(3,is_prior=True);wire=json.loads(messages_for([p])[1]['content']);short,table=api.encode_wire(wire)
        self.assertEqual(short['pairs'][0]['right']['evidence_quotes'],[
            dict(wire['pairs'][0]['right']['evidence_quotes'][0],evidence_id='R0')])
        response=self.answer(short);row=response['results'][0]
        row.update(necessary_scope_consistent=True,definition_identity_sufficient=True)
        decoded=api.decode_response(response,table)['results'][0]
        self.assertTrue(core._validate(p,core._anonymous(p),decoded,False)['eligible_identity_edge'])
        self.assertEqual(short['pairs'][0]['right']['evidence_quotes'][0]['provenance'],'prior_generated_definition')

    def test_mapping_tampering_and_wire_text_mutation_fail_roundtrip_verification(self):
        api=self.api();wire=source.anonymous_packet(packet(3));short,table=api.encode_wire(wire)
        changed=copy.deepcopy(short);changed['left']['scope']='different scope'
        with self.assertRaises(ValueError):api.restore_wire(changed,table)
        changed=copy.deepcopy(table);changed['pairs']['P0']['candidate_id']='changed'
        with self.assertRaises(ValueError):api.decode_response(self.answer(short),changed)


if __name__=='__main__':unittest.main()
