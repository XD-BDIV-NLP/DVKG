"""原生图内链接预测：独立于公共参考事实协议，保留开放关系。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

PROTOCOL = 'native_graph_pair_grouped_transductive_v1'
SEEDS = (42, 43, 44)
METRICS = ('mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def graph_triples(document):
    """保留原生谓词；无谓词图仅标为邻接探针，不从描述编造关系。"""
    graph = document.get('native_graph', document)
    rows = graph.get('edges', graph.get('relations', []))
    triples = []
    untyped = 0
    for edge in rows:
        h = edge.get('head_entity_id', edge.get('source', edge.get('head')))
        t = edge.get('tail_entity_id', edge.get('target', edge.get('tail')))
        r = next((edge[k] for k in ('predicate','relation','type')
                  if isinstance(edge.get(k), str) and edge[k].strip()), None)
        if r is None:
            r = '__native_related__'
            untyped += 1
        if not isinstance(h, str) or not isinstance(t, str) or not h or not t:
            raise ValueError('Native graph edge has no explicit endpoint IDs')
        triples.append((h,r,t))
        if graph.get('directed') is False and h != t:
            triples.append((t,r,h))
    if not triples:
        raise ValueError('No native graph edges')
    return dict(triples=triples, relation_semantics=(
        'adjacency_only_generic_relation' if untyped == len(rows) else
        'mixed_native_and_generic' if untyped else 'native_predicates'),
        native_edges=len(rows), untyped_edges=untyped,
        reciprocal_expansion=graph.get('directed') is False)


def prepare_native_graph(triples, seed=42):
    """同实体对的所有方向与关系同折；保持留出实体和关系可传导。"""
    clean = set()
    original_count = 0
    for triple in triples:
        original_count += 1
        if (not isinstance(triple, (tuple, list)) or len(triple) != 3 or
                any(not isinstance(v, str) or not v.strip() for v in triple)):
            raise ValueError('Expected nonempty native head/relation/tail strings')
        h, r, t = (v.strip() for v in triple)
        clean.add((digest(['entity', h]), digest(['relation', r]), digest(['entity', t])))
    if not clean:
        raise ValueError('No native triples')
    groups = defaultdict(list)
    for h, r, t in sorted(clean):
        groups[tuple(sorted((h, t)))].append([h, r, t])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    held = max(1, int(n * .1)) if n >= 10 else 0
    assignments = {k: 'train' for k in keys}
    for k in keys[:held]:
        assignments[k] = 'test'
    for k in keys[held:2 * held]:
        assignments[k] = 'validation'
    train = [v for k in keys if assignments[k] == 'train' for v in groups[k]]
    entities = {e for h, r, t in train for e in (h, t)}
    relations = {r for h, r, t in train}
    moved = 0
    for k in keys:
        if assignments[k] == 'train':
            continue
        if any(h not in entities or t not in entities or r not in relations for h, r, t in groups[k]):
            assignments[k] = 'train'
            moved += 1
            for h, r, t in groups[k]:
                entities.update((h, t))
                relations.add(r)
    splits = {name: sorted(v for k in keys if assignments[k] == name for v in groups[k])
              for name in ('train', 'validation', 'test')}
    counts = dict(original_triples=original_count, unique_triples=len(clean),
                  duplicates=original_count-len(clean), entities=len({e for h, r, t in clean for e in (h, t)}),
                  relations=len({r for h, r, t in clean}), pair_groups=n,
                  coverage_groups_moved_to_train=moved,
                  **{name: len(values) for name, values in splits.items()})
    value = dict(protocol=PROTOCOL, split_seed=seed, counts=counts, splits=splits,
                 status='ready' if all(splits.values()) else 'insufficient_holdout',
                 disclosure='native graph reconstruction; method-specific facts, vocabulary and candidate size; not a shared-gold quality ranking',
                 raw_text_exported=False)
    value['bundle_id'] = digest(value)
    return value


def validate_bundle(value):
    if value['bundle_id'] != digest({k:v for k,v in value.items() if k != 'bundle_id'}):
        raise ValueError('Frozen native split changed')
    folds = {k: {tuple(t) for t in v} for k, v in value['splits'].items()}
    if any(folds[a] & folds[b] for a, b in [('train','test'),('train','validation'),('test','validation')]):
        raise ValueError('Native triple leakage')
    pair_folds = {}
    for name, triples in folds.items():
        for h, r, t in triples:
            if any(len(v) != 64 or set(v)-set('0123456789abcdef') for v in (h,r,t)):
                raise ValueError('GPU payload must contain opaque graph identifiers only')
            pair = tuple(sorted((h, t)))
            if pair in pair_folds and pair_folds[pair] != name:
                raise ValueError('Reverse or parallel relation leakage')
            pair_folds[pair] = name
    train_entities = {v for h,r,t in folds['train'] for v in (h,t)}
    train_relations = {r for h,r,t in folds['train']}
    if any(h not in train_entities or t not in train_entities or r not in train_relations
           for h,r,t in folds['test'] | folds['validation']):
        raise ValueError('Unseen transductive endpoint or relation')
    return value


def summarize_seeds(rows):
    if {r.get('seed') for r in rows} != set(SEEDS) or len(rows) != 3 or any(r.get('status') != 'completed' for r in rows):
        return dict(status='partial', completed=sum(r.get('status') == 'completed' for r in rows))
    return dict(status='completed', seeds=list(SEEDS), metrics={k:dict(
        mean=statistics.mean(r['metrics'][k] for r in rows),
        std=statistics.stdev(r['metrics'][k] for r in rows)) for k in METRICS})


def train(bundle_path, output, model_name, seed, device, *, epochs=1000, deadline=None):
    """使用已安装PyKEEN原生实现训练，不调用构图或模型API。"""
    import numpy as np
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    from pykeen.training.callbacks import TrainingCallback

    if model_name not in ('TransE', 'TransR') or seed not in SEEDS or epochs not in (2, 1000):
        raise ValueError('Unregistered native KGE model/seed/epochs')
    bundle = validate_bundle(json.loads(Path(bundle_path).read_text()))
    if bundle['status'] != 'ready':
        raise ValueError('Native graph has no valid heldout transductive split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol=PROTOCOL, bundle_id=bundle['bundle_id'], model=model_name,
                    seed=seed, epochs=epochs, dimension=128, relation_dimension=128,
                    batch_size=256, negatives=16, learning_rate=.001,
                    optimizer='Adam', loss='MarginRankingLoss', margin=1.0,
                    scoring_fct_norm=1, evaluator='filtered_both_sides_realistic',
                    selection='fixed_final_epoch; no test-dependent tuning')
    identity['job_id'] = digest(identity)
    result_path = output/'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['identity'] != identity or not (output/'model/trained_model.pkl').exists():
            raise ValueError('Existing native result identity or model missing')
        return previous
    (output/'contract.json').write_text(json.dumps(identity, indent=2))
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    training = TriplesFactory.from_labeled_triples(np.asarray(bundle['splits']['train']),
                                                  create_inverse_triples=False)
    factories = {'train':training}
    for name in ('validation','test'):
        factories[name] = TriplesFactory.from_labeled_triples(np.asarray(bundle['splits'][name]),
            entity_to_id=training.entity_to_id, relation_to_id=training.relation_to_id,
            create_inverse_triples=False)

    class Progress(TrainingCallback):
        def post_epoch(self, epoch, epoch_loss, **kwargs):
            p = dict(epoch=epoch, loss=float(epoch_loss), observed_unix=time.time(),
                     epochs=epochs, job_id=identity['job_id'])
            temp = output/'progress.tmp'
            temp.write_text(json.dumps(p))
            temp.replace(output/'progress.json')
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError('Original native KGE execution deadline reached')

    started = time.time()
    model_kwargs = dict(embedding_dim=128, scoring_fct_norm=1)
    if model_name == 'TransR':
        model_kwargs['relation_dim'] = 128
    result = pipeline(training=training, validation=factories['validation'], testing=factories['test'],
        model=model_name, model_kwargs=model_kwargs, loss='MarginRankingLoss', loss_kwargs=dict(margin=1.0),
        optimizer='Adam', optimizer_kwargs=dict(lr=.001), training_loop='sLCWA',
        negative_sampler='basic', negative_sampler_kwargs=dict(num_negs_per_pos=16, filtered=True),
        training_kwargs=dict(num_epochs=epochs, batch_size=256, use_tqdm=False, callbacks=[Progress()],
                             checkpoint_directory=str(output/'checkpoints'), checkpoint_name='training.pt',
                             checkpoint_frequency=1, checkpoint_on_failure=True),
        evaluator='RankBasedEvaluator', evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=32, slice_size=512, use_tqdm=False),
        random_seed=seed, device=device)
    result.save_to_directory(output/'model')
    keys={'mrr':'inverse_harmonic_mean_rank', 'hits_at_1':'hits_at_1',
          'hits_at_3':'hits_at_3', 'hits_at_10':'hits_at_10'}
    metrics={k:float(result.metric_results.get_metric('both.realistic.'+v)) for k,v in keys.items()}
    if len(result.losses) != epochs or any(not math.isfinite(v) for v in metrics.values()):
        raise ValueError('Native training did not finish requested epochs or valid ranking')
    value=dict(status='completed', model=model_name, seed=seed, identity=identity, metrics=metrics,
               counts=bundle['counts'], actual_epochs=len(result.losses),
               elapsed_seconds=time.time()-started, started_unix=started, completed_unix=time.time(),
               model_sha256=hashlib.sha256((output/'model/trained_model.pkl').read_bytes()).hexdigest(),
               raw_text_received=False, model_API_calls=0)
    result_path.write_text(json.dumps(value, indent=2, allow_nan=False))
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['TransE','TransR'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deadline', type=float)
    args=parser.parse_args()
    print(json.dumps(train(args.bundle,args.output,args.model,args.seed,args.device,
                           epochs=args.epochs, deadline=args.deadline)))


if __name__ == '__main__':
    main()
