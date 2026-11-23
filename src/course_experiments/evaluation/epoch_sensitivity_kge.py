"""固定种子42，从既有1000轮完整检查点续到2000/3000轮。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time


def target_epochs(smoke=False):
    return (1001,1002) if smoke else (2000,3000)


def validate_checkpoint(checkpoint,entity_to_id,relation_to_id):
    required={'random_state','np_random_state','torch_random_state','torch_cuda_random_state'}
    optimizer=checkpoint.get('optimizer_state_dict',{})
    if (checkpoint.get('epoch')!=1000 or len(checkpoint.get('loss',[]))!=1000 or
        checkpoint.get('random_seed')!=42 or not checkpoint.get('model_state_dict') or
        not optimizer.get('state') or not optimizer.get('param_groups') or
        not required<=set(checkpoint) or
        checkpoint.get('entity_to_id_dict')!=entity_to_id or
        checkpoint.get('relation_to_id_dict')!=relation_to_id):
        raise ValueError('Incomplete or mismatched original 1000-epoch checkpoint')
    return 1000


def validate_loss_prefix(losses,original,epoch):
    if len(losses)!=epoch or losses[:1000]!=original:
        raise ValueError('Continued loss history does not preserve the original 1000 epochs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)


def run(job_path,output,device,deadline,*,smoke=False):
    import numpy as np
    import torch
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop
    from pykeen.training.callbacks import TrainingCallback
    from pykeen.triples import TriplesFactory

    spec=importlib.util.spec_from_file_location('frozen_native_core',Path(__file__).with_name('native_graph_kge.py'))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    job=json.loads(Path(job_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    baseline_path=Path(job['baseline_result']['path'])
    assert sha(baseline_path)==job['baseline_result']['sha256']
    baseline=json.loads(baseline_path.read_text());old=baseline_path.parent
    assert baseline['seed']==42 and baseline['actual_epochs']==1000 and baseline['status']=='completed'
    assert sha(old/'model/trained_model.pkl')==baseline['model_sha256']
    assert sha(job['bundle']['path'])==job['bundle']['sha256']
    bundle=core.validate_bundle(json.loads(Path(job['bundle']['path']).read_text()))
    assert bundle['bundle_id']==baseline['identity']['bundle_id']
    training=TriplesFactory.from_labeled_triples(np.asarray(bundle['splits']['train']),create_inverse_triples=False)
    factories={'train':training}
    for name in ['validation','test']:
        factories[name]=TriplesFactory.from_labeled_triples(np.asarray(bundle['splits'][name]),
            entity_to_id=training.entity_to_id,relation_to_id=training.relation_to_id,create_inverse_triples=False)
    source_cp=old/'checkpoints/training.pt'
    assert sha(source_cp)==job['baseline_checkpoint']['sha256']
    checkpoint=torch.load(source_cp,map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,training.entity_to_id,training.relation_to_id)
    original=json.loads((old/'model/results.json').read_text())['losses']
    validate_loss_prefix(checkpoint['loss'],original,1000)
    model=torch.load(old/'model/trained_model.pkl',map_location='cpu',weights_only=False)
    assert set(model.state_dict())==set(checkpoint['model_state_dict'])
    assert all(torch.equal(v,checkpoint['model_state_dict'][k]) for k,v in model.state_dict().items())
    targets=target_epochs(smoke)
    contract=dict(protocol='native_graph_epoch_sensitivity_seed42_v1',name=job['name'],model=baseline['model'],
        seed=42,split_seed=42,targets=list(targets),smoke=smoke,baseline_job_id=baseline['identity']['job_id'],
        baseline_model_sha256=baseline['model_sha256'],baseline_checkpoint=job['baseline_checkpoint'],
        bundle_id=bundle['bundle_id'],parameters=baseline['identity'],
        selection='fixed epochs, validation and test reported separately; no test-dependent stopping')
    contract['job_id']=core.digest(contract)
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    else:atomic(out/'contract.json',contract)
    if all((out/f'epoch_{epoch}/result.json').exists() for epoch in targets):
        return dict(status='completed',targets=list(targets),job_id=contract['job_id'])
    cp_dir=out/'checkpoints';cp_dir.mkdir(exist_ok=True)
    if not (cp_dir/'training.pt').exists():
        torch.save(checkpoint,cp_dir/'training.pt')
    del checkpoint
    torch.set_num_threads(2);torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=model.to(device)
    loop=SLCWATrainingLoop(model=model,triples_factory=training,optimizer='Adam',optimizer_kwargs=dict(lr=.001),
        negative_sampler='basic',negative_sampler_kwargs=dict(num_negs_per_pos=16,filtered=True))
    saved_cp=torch.load(cp_dir/'training.pt',map_location='cpu',weights_only=False)
    assert loop.checksum==saved_cp['checksum'],'Original model/Adam settings do not match'
    assert saved_cp['epoch']>=1000
    del saved_cp
    started=time.time()

    def evaluate(epoch):
        dest=out/f'epoch_{epoch}';dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():
            value=json.loads((dest/'result.json').read_text());assert value['contract_id']==contract['job_id']
            return
        validate_loss_prefix(loop.losses_per_epochs,original,epoch)
        # 保存评测前的随机状态，评测不能改变后续训练抽样。
        state=(random.getstate(),np.random.get_state(),torch.random.get_rng_state(),torch.cuda.get_rng_state())
        was_training=model.training
        loop._save_state(dest/'training.pt',triples_factory=training)
        torch.save(model,dest/'trained_model.pkl')
        atomic(dest/'losses.json',loop.losses_per_epochs)
        metrics={}
        try:
            model.eval()
            for split in ['validation','test']:
                filters=[training.mapped_triples]
                if split=='test':filters.append(factories['validation'].mapped_triples)
                evaluator=RankBasedEvaluator(filtered=True)
                result=evaluator.evaluate(model=model,mapped_triples=factories[split].mapped_triples,
                    additional_filter_triples=filters,batch_size=32,slice_size=512,use_tqdm=False)
                keys={'mrr':'inverse_harmonic_mean_rank','hits_at_1':'hits_at_1','hits_at_3':'hits_at_3','hits_at_10':'hits_at_10'}
                metrics[split]={k:float(result.get_metric('both.realistic.'+v)) for k,v in keys.items()}
        finally:
            random.setstate(state[0]);np.random.set_state(state[1]);torch.random.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3])
            model.train(was_training)
        value=dict(status='completed',name=job['name'],model=baseline['model'],seed=42,actual_epochs=epoch,
            contract_id=contract['job_id'],baseline_job_id=baseline['identity']['job_id'],bundle_id=bundle['bundle_id'],
            checkpoint_sha256=sha(dest/'training.pt'),model_sha256=sha(dest/'trained_model.pkl'),
            losses_sha256=sha(dest/'losses.json'),metrics=metrics['test'],validation_metrics=metrics['validation'],
            original_1000_loss_prefix_verified=True,optimizer_resumed=True,counts=bundle['counts'],
            elapsed_seconds=time.time()-started,completed_unix=time.time(),model_API_calls=0,smoke=smoke)
        atomic(dest/'result.json',value)

    class Milestones(TrainingCallback):
        def post_epoch(self,epoch,epoch_loss,**kwargs):
            atomic(out/'progress.json',dict(epoch=epoch,loss=float(epoch_loss),targets=list(targets),observed_unix=time.time(),job_id=contract['job_id']))
            if epoch in targets:evaluate(epoch)
            if time.time()>=deadline:
                loop._save_state(cp_dir/'training.pt',triples_factory=training)
                raise RuntimeError('Original KGE deadline reached; current progress preserved')

    loop.train(triples_factory=training,num_epochs=targets[-1],batch_size=256,use_tqdm=False,
        checkpoint_directory=cp_dir,checkpoint_name='training.pt',checkpoint_frequency=1,
        checkpoint_on_failure=True,callbacks=[Milestones()])
    validate_loss_prefix(loop.losses_per_epochs,original,targets[-1])
    assert all((out/f'epoch_{epoch}/result.json').exists() for epoch in targets)
    value=dict(status='completed',targets=list(targets),job_id=contract['job_id'],actual_epochs=targets[-1])
    atomic(out/'summary.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.job,a.output,a.device,a.deadline,smoke=a.smoke)))
