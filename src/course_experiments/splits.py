from __future__ import annotations
import unicodedata
import copy
from .manifest import digest

RELATIONS={'is_a','part_of','prerequisite_of','causes','affects','used_for','consists_of','compared_with','instance_of','evaluated_by','implemented_by','derived_from'}

def norm(value):
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split())

def entity_lookup(entities):
    lookup={}
    for e in entities:
        for name in [e['canonical_name'],*e.get('aliases',[])]:
            lookup.setdefault(norm(name),set()).add((str(e.get('scope','')),e['entity_id']))
        lookup.setdefault(norm(e['entity_id']),set()).add((str(e.get('scope','')),e['entity_id']))
    return lookup

def resolve(value,scope,lookup):
    rows=lookup.get(norm(value),set())
    if scope: rows={r for r in rows if r[0]==str(scope)}
    return next(iter(rows))[1] if len(rows)==1 else None

def canonical(head,relation,tail):
    if relation=='consists_of':head,relation,tail=tail,'part_of',head
    if relation=='compared_with':head,tail=sorted([head,tail])
    return (head,relation,tail)

def canonical_reference(reference):
    result=copy.deepcopy(reference);rows=result['entities'];parents={e['entity_id']:e['entity_id'] for e in rows};by_name={}
    if len(parents)!=len(rows):raise ValueError('Duplicate independent reference entity IDs')
    def find(n):
        while parents[n]!=n:
            parents[n]=parents[parents[n]];n=parents[n]
        return n
    for e in rows:
        for name in [e['canonical_name'],*e.get('aliases',[])]:
            key=(str(e.get('scope','')),norm(name))
            if key in by_name:
                a,b=find(e['entity_id']),find(by_name[key]);parents[max(a,b)]=min(a,b)
            else:by_name[key]=e['entity_id']
    mapping={n:find(n) for n in parents};merged={}
    for e in sorted(rows,key=lambda x:x['entity_id']):
        target=mapping[e['entity_id']]
        if target not in merged:merged[target]=dict(e,entity_id=target,aliases=[],citations=[],source_entity_ids=[])
        row=merged[target];row['source_entity_ids'].append(e['entity_id'])
        row['aliases']+= [e['canonical_name'],*e.get('aliases',[])];row['citations']+=e.get('citations',[])
    for e in merged.values():
        e['aliases']=sorted(set(a for a in e['aliases'] if norm(a)!=norm(e['canonical_name'])))
        e['citations']=list({digest(c):c for c in e['citations']}.values())
    result['entities']=list(merged.values());lookup=entity_lookup(result['entities'])
    for f in result['facts']:
        for endpoint in ('head','tail'):
            old=f[endpoint];new=mapping.get(old) or resolve(old,f.get('scope',''),lookup)
            if new is None:raise ValueError('Reference endpoint ambiguous after alias canonicalization')
            f[endpoint]=new
    result['reference_alias_mapping']=mapping
    return result

def freeze_split(reference,seed=42,ratios=(.8,.1,.1)):
    if seed!=42 or tuple(ratios)!=(.8,.1,.1):raise ValueError('Fixed seed/ratios required')
    canonical_ref=canonical_reference(reference)
    lookup=entity_lookup(canonical_ref['entities']);groups={};ids={e['entity_id'] for e in canonical_ref['entities']}
    for f in canonical_ref['facts']:
        h=f['head'] if f['head'] in ids else resolve(f['head'],f.get('scope',''),lookup)
        t=f['tail'] if f['tail'] in ids else resolve(f['tail'],f.get('scope',''),lookup)
        if h is None or t is None or f['relation'] not in RELATIONS:raise ValueError('Reference fact unresolved/unknown predicate')
        triple=canonical(h,f['relation'],t);key=digest(triple)
        groups.setdefault(key,dict(triple=list(triple),fact_ids=[],conditions=[]))
        groups[key]['fact_ids'].append(f['fact_id']);groups[key]['conditions'].append(f.get('conditions',''))
    ordered=sorted(groups,key=lambda g:digest([seed,g]));n=len(ordered)
    train=int(n*.8);valid=int(n*.1)
    folds=dict(train=ordered[:train],validation=ordered[train:train+valid],test=ordered[train+valid:])
    payload=dict(course_id=reference['course_id'],reference_id=digest(reference),reference_alias_mapping=canonical_ref['reference_alias_mapping'],seed=seed,ratios=list(ratios),groups=groups,folds=folds,
        policy='NFKC-casefold scoped explicit aliases; consists_of reversed to part_of; compared_with sorted; conditional variants collapse to plain triple')
    payload['split_id']=digest(payload);validate_split(payload);return payload

def validate_split(split):
    if split.get('split_id')!=digest({k:v for k,v in split.items() if k!='split_id'}):raise ValueError('Split hash mismatch')
    folds=[set(split['folds'][k]) for k in ('train','validation','test')]
    if any(folds[i]&folds[j] for i in range(3) for j in range(i)):raise ValueError('Fold leakage')
    if set.union(*folds)!=set(split['groups']):raise ValueError('Incomplete fold assignment')
    if len({tuple(g['triple']) for g in split['groups'].values()})!=len(split['groups']):raise ValueError('Equivalent triples in separate groups')
    return split

def project_graph(graph,reference,split):
    validate_split(split)
    if split['reference_id']!=digest(reference):raise ValueError('Reference changed after splitting')
    canonical_ref=canonical_reference(reference)
    if split['reference_alias_mapping']!=canonical_ref['reference_alias_mapping']:raise ValueError('Reference alias components differ from frozen split')
    lookup=entity_lookup(canonical_ref['entities']);mapping={};audit=dict(unmatched_entities=[],excluded_relations=[],heldout_removed=[],mapping=[],reference_alias_mapping=canonical_ref['reference_alias_mapping'])
    for e in graph['entities']:
        identities=e.get('identity_provenance') or [e]
        scopes={str(i.get('scope','')) for i in identities if i.get('scope')}
        if len(scopes)>1:
            audit['unmatched_entities'].append(e['entity_id']);continue
        if scopes:identities=[i for i in identities if i.get('scope')]
        choices={resolve(n,i.get('scope',''),lookup) for i in identities
                 for n in [i['canonical_name'],*i.get('aliases',[])]}-{None}
        if len(choices)!=1:audit['unmatched_entities'].append(e['entity_id']);continue
        mapped=next(iter(choices));mapping[e['entity_id']]=mapped
        audit['mapping'].append(dict(graph_entity_id=e['entity_id'],reference_entity_id=mapped,policy='unique scoped explicit name/alias'))
    heldout=set(split['folds']['validation']+split['folds']['test']);training=set()
    for i,f in enumerate(graph['relations']):
        h=mapping.get(f.get('head_entity_id'));t=mapping.get(f.get('tail_entity_id'));r=f.get('relation')
        if not h or not t or r not in RELATIONS:audit['excluded_relations'].append(f.get('relation_id',str(i)));continue
        triple=canonical(h,r,t);key=digest(triple)
        if key in heldout:audit['heldout_removed'].append(f.get('relation_id',str(i)));continue
        training.add(triple)
    vocabulary=sorted(e['entity_id'] for e in canonical_ref['entities']);relations=sorted(RELATIONS-{'consists_of'})
    def fold(k):return [split['groups'][g]['triple'] for g in split['folds'][k]]
    trained={x for h,r,t in training for x in (h,t)};trained_relations={r for h,r,t in training}
    result=dict(course_id=reference['course_id'],graph_id=digest(graph),split_id=split['split_id'],reference_id=split['reference_id'],training=[list(t) for t in sorted(training)],validation=fold('validation'),test=fold('test'),test_ids=split['folds']['test'],validation_ids=split['folds']['validation'],entity_vocabulary=vocabulary,relation_vocabulary=relations,audit=audit,
        covered_test_mask=[t[0] in trained and t[2] in trained and t[1] in trained_relations for t in fold('test')],covered_validation_mask=[t[0] in trained and t[2] in trained and t[1] in trained_relations for t in fold('validation')],condition_projection=split['policy'])
    result['projection_id']=digest(result);return result

def common_test_mask(projections):
    if not projections:raise ValueError('No projections')
    if any(p['split_id']!=projections[0]['split_id'] or p['test']!=projections[0]['test'] for p in projections):raise ValueError('Different fixed test sets')
    return [all(p['covered_test_mask'][i] for p in projections) for i in range(len(projections[0]['test']))]
