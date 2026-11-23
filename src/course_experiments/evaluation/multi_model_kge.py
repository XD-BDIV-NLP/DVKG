from __future__ import annotations

import difflib
import hashlib

from ..manifest import digest

MODELS=('TransR','DistMult','ComplEx','RotatE','TransE')
SEEDS=(42,43,44)
EPOCHS=(500,1000)
NATIVE_SHAS={'6929d67b0ebd9c06e7ac872de6e54f2cb61d14a5a1182205c27557a26ce988f3',
             '8dfc7d491b6e377df384763358e9a62f4d6db23ab4fa510677de03bafe3cfa9e'}


def model_kwargs(model):
    if model not in MODELS:raise ValueError('Model not frozen')
    return dict(embedding_dim=128,**({'relation_dim':128} if model=='TransR' else {}))


def derive_native(raw,model,seed):
    source_sha=hashlib.sha256(raw).hexdigest()
    if source_sha not in NATIVE_SHAS or model not in MODELS or seed not in SEEDS:raise ValueError('Unreviewed native source/model/seed')
    new=raw
    def once(old,target):
        nonlocal new
        if new.count(old)!=1:raise ValueError('Frozen adapter anchor differs')
        new=new.replace(old,target)
    once(b"model='TransE',epochs=",("model="+repr(model)+",epochs=").encode())
    once(b'seed=42)',('seed='+str(seed)+')').encode())
    once(b'args.seed!=42',('args.seed!='+str(seed)).encode())
    if model=='TransR':
        once(b"model_kwargs={'embedding_dim':budget['dimension']}",
             b"model_kwargs={'embedding_dim':budget['dimension'],'relation_dim':budget['dimension']}")
    changed=sum(a!=b for a,b in zip(raw.splitlines(),new.splitlines()))
    if changed>3 or len(raw.splitlines())!=len(new.splitlines()):raise ValueError('Unexpected model adapter diff')
    return new,dict(original_sha256=source_sha,private_sha256=hashlib.sha256(new).hexdigest(),model=model,
                    training_seed=seed,split_seed=42,model_kwargs=model_kwargs(model),changed_lines=changed,
                    diff=''.join(difflib.unified_diff(raw.decode().splitlines(True),new.decode().splitlines(True),fromfile='frozen/kge.py',tofile='private/kge.py')))


def training_signature(projection,model,seed,epochs,software):
    return digest(dict(training=sorted(projection['training']),entities=projection['entity_vocabulary'],relations=projection['relation_vocabulary'],
                       model=model,model_kwargs=model_kwargs(model),seed=seed,epochs=epochs,software=software,
                       batch_size=256,negative_samples=16,learning_rate=.001,
                       training_algorithm='existing_pykeen_pipeline_native_model_defaults'))


def matrix(members):
    jobs=[]
    for seed in SEEDS:
        for model in MODELS:
            for cohort in members:
                for member in members[cohort]:
                    for epochs in EPOCHS:
                        name=f'{cohort}_{member}_{model}_s{seed}_e{epochs}'
                        jobs.append(dict(name=name,cohort=cohort,member=member,model=model,seed=seed,epochs=epochs))
    return jobs
