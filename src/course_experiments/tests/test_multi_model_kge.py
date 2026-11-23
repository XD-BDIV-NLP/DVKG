import copy
import hashlib
import unittest
from pathlib import Path

try:
    from course_experiments.evaluation import multi_model_kge as multi
except ImportError:
    multi=None


class MultiModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(multi,'Multi-model adapter not implemented')

    def native(self,sha='8dfc7d491b6e377df384763358e9a62f4d6db23ab4fa510677de03bafe3cfa9e'):
        return (Path(__file__).parent/'fixtures/kge_native'/(sha+'.py')).read_bytes()

    def test_transr_relation_dimension_is_explicit(self):
        self.assertEqual(multi.model_kwargs('TransR'),dict(embedding_dim=128,relation_dim=128))
        self.assertEqual(multi.model_kwargs('ComplEx'),dict(embedding_dim=128))

    def test_private_source_only_changes_allowed_model_seed_and_kwargs(self):
        raw=self.native()
        for model in multi.MODELS:
            for seed in multi.SEEDS:
                changed,proof=multi.derive_native(raw,model,seed)
                self.assertEqual(proof['original_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(proof['training_seed'],seed)
                self.assertEqual(proof['split_seed'],42)
                self.assertLessEqual(proof['changed_lines'],3)
                compile(changed,'private_kge','exec')
        self.assertEqual(multi.derive_native(raw,'TransE',42)[0],raw)

    def test_unknown_code_model_and_seed_are_rejected(self):
        for raw,model,seed in [(self.native()+b'\n','TransR',42),(self.native(),'ConvE',42),(self.native(),'TransE',99)]:
            with self.assertRaises(ValueError):multi.derive_native(raw,model,seed)

    def test_both_frozen_native_versions_are_supported(self):
        for sha in multi.NATIVE_SHAS:
            raw=self.native(sha)
            for model in multi.MODELS:
                for seed in multi.SEEDS:
                    derived,proof=multi.derive_native(raw,model,seed)
                    compile(derived,'frozen_private','exec')
                    self.assertEqual(proof['original_sha256'],sha)

    def test_signature_deduplicates_only_identical_training(self):
        projection=dict(training=[['a','r','b']],entity_vocabulary=['a','b'],relation_vocabulary=['r'],covered_test_mask=[True])
        kwargs=dict(model='TransR',seed=42,epochs=500,software=dict(torch='2.5.1',pykeen='1.11.1'))
        first=multi.training_signature(projection,**kwargs)
        other=copy.deepcopy(projection);other['covered_test_mask']=[False]
        self.assertEqual(first,multi.training_signature(other,**kwargs))
        for field,value in [('seed',43),('epochs',1000),('model','DistMult')]:
            args=dict(kwargs,**{field:value});self.assertNotEqual(first,multi.training_signature(projection,**args))
        other['training']=[['b','r','a']]
        self.assertNotEqual(first,multi.training_signature(other,**kwargs))

    def test_matrix_contains_all_five_models_including_transr(self):
        jobs=multi.matrix({'compAB':['fused','textbook_only'],'compBookCount':['AB','ABCD']})
        self.assertEqual(len(jobs),120)
        self.assertEqual({j['model'] for j in jobs},set(multi.MODELS))
        self.assertEqual(len({j['name'] for j in jobs}),120)
        self.assertEqual(jobs[0]['model'],'TransR')


if __name__=='__main__':unittest.main()
