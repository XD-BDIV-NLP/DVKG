"""续训检查点和固定轮数身份不能混用。"""
import importlib
import unittest


class EpochSensitivityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.epoch_sensitivity_kge'),
                             '轮数敏感性续训尚未实现')
        return importlib.import_module('course_experiments.evaluation.epoch_sensitivity_kge')

    def checkpoint(self):
        return dict(epoch=1000,loss=[.5]*1000,model_state_dict={'weight':1},
            optimizer_state_dict={'state':{'x':1},'param_groups':[{'lr':.001}]},
            random_state=1,np_random_state=1,torch_random_state=1,torch_cuda_random_state=1,
            random_seed=42,entity_to_id_dict={'a':0},relation_to_id_dict={'r':0})

    def test_complete_checkpoint_preserves_optimizer_and_losses(self):
        m=self.module();c=self.checkpoint()
        self.assertEqual(m.validate_checkpoint(c,{'a':0},{'r':0}),1000)

    def test_missing_optimizer_wrong_epoch_or_mapping_is_rejected(self):
        m=self.module()
        for key,value in [('epoch',999),('loss',[.5]*999),('random_seed',43),
                          ('optimizer_state_dict',{}),('entity_to_id_dict',{'a':1})]:
            c=self.checkpoint();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.validate_checkpoint(c,{'a':0},{'r':0})

    def test_missing_rng_state_is_rejected(self):
        m=self.module();c=self.checkpoint();del c['torch_random_state']
        with self.assertRaises(ValueError):m.validate_checkpoint(c,{'a':0},{'r':0})

    def test_fixed_epochs_and_smoke_are_different_contracts(self):
        m=self.module()
        self.assertEqual(m.target_epochs(False),(2000,3000))
        self.assertEqual(m.target_epochs(True),(1001,1002))

    def test_training_loss_prefix_must_match_original(self):
        m=self.module()
        m.validate_loss_prefix([.5]*1000+[.4]*1000,[.5]*1000,2000)
        with self.assertRaises(ValueError):m.validate_loss_prefix([.4]*2000,[.5]*1000,2000)
        with self.assertRaises(ValueError):m.validate_loss_prefix([.5]*1999,[.5]*1000,2000)


if __name__=='__main__':unittest.main()
