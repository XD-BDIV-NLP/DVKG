"""新版本身份wire的显式短标签双射；不改定义、scope或逐字引句。"""
from __future__ import annotations

import copy

from ..manifest import digest


POLICY=dict(version='identity_explicit_short_wire_labels_v1',candidate_labels='P0..P7',batch_label='B0',
    evidence_labels='global_left_L0.._right_R0.._exact_pair_side_membership_required',
    restoration='declared_bijection_only_no_prefix_hash_guess_or_invalid_citation_removal',
    text='all_scope_definitions_names_quotes_provenance_unchanged',
    validation='restore_original_ids_then_unchanged_original_strict_validator',
    old_results='retained_separate_request_configuration_not_reclassified',
    human_audited=False,model_calls=0)
INSTRUCTION='标识字段使用本请求显式给出的短标签；candidate_id、batch_id和evidence_id均须逐字符返回对应标签。不要重算hash、猜测标签或引用其它pair/其它侧的标签。原身份判定准则与逐字引文规则不变。'


def encode_wire(wire):
    result=copy.deepcopy(wire);batched='pairs' in wire
    originals=wire['pairs'] if batched else [wire]
    encoded=result['pairs'] if batched else [result]
    if not 1<=len(originals)<=8:raise ValueError('Wire requires one to eight pairs')
    if len({p['candidate_id'] for p in originals})!=len(originals):raise ValueError('Duplicate original wire candidate')
    table=dict(policy=POLICY,batched=batched,wire_digest=digest(wire),pairs={})
    if batched:table['batch_id']=wire['batch_id'];result['batch_id']='B0'
    counts=dict(left=0,right=0)
    for number,(old,new) in enumerate(zip(originals,encoded)):
        label='P'+str(number);new['candidate_id']=label;row=dict(candidate_id=old['candidate_id'],sides={})
        for side,prefix in [('left','L'),('right','R')]:
            fields=[name for name in ['source_quotes','evidence_quotes','definition_quotes'] if name in old[side]]
            if len(fields)!=1:raise ValueError('Exactly one declared quote collection per wire side required')
            field=fields[0];inverse={};labels={}
            for original,short in zip(old[side][field],new[side][field]):
                eid=original['evidence_id']
                if eid not in inverse:
                    token=prefix+str(counts[side]);counts[side]+=1;inverse[eid]=token;labels[token]=eid
                short['evidence_id']=inverse[eid]
            row['sides'][side]=dict(field=field,labels=labels)
        table['pairs'][label]=row
    table['mapping_id']=digest(table)
    return result,table


def _check_table(table):
    if table.get('mapping_id')!=digest({k:v for k,v in table.items() if k!='mapping_id'}):
        raise ValueError('Frozen wire label mapping changed')


def restore_wire(short,table):
    _check_table(table);result=copy.deepcopy(short)
    pairs=result['pairs'] if table['batched'] else [result]
    if table['batched']:
        if result.get('batch_id')!='B0':raise ValueError('Unknown batch label')
        result['batch_id']=table['batch_id']
    if [p['candidate_id'] for p in pairs]!=list(table['pairs']):raise ValueError('Wire candidate labels changed')
    for pair in pairs:
        row=table['pairs'][pair['candidate_id']];pair['candidate_id']=row['candidate_id']
        for side,record in row['sides'].items():
            for quote in pair[side][record['field']]:
                if quote.get('evidence_id') not in record['labels']:raise ValueError('Unknown or cross-side evidence label')
                quote['evidence_id']=record['labels'][quote['evidence_id']]
    if digest(result)!=table['wire_digest']:raise ValueError('Wire text or other original fields changed')
    return result


def decode_response(response,table):
    _check_table(table);result=copy.deepcopy(response)
    if not isinstance(result,dict):raise ValueError('Identity label response must be an object')
    if table['batched']:
        if result.get('batch_id')!='B0' or not isinstance(result.get('results'),list):raise ValueError('Unknown batch label or response shape')
        result['batch_id']=table['batch_id'];pairs=result['results']
    else:pairs=[result]
    labels=[p.get('candidate_id') for p in pairs]
    if len(set(labels))!=len(labels) or set(labels)!=set(table['pairs']):raise ValueError('Unknown missing or duplicate candidate labels')
    for pair in pairs:
        row=table['pairs'][pair['candidate_id']];pair['candidate_id']=row['candidate_id']
        for side,record in row['sides'].items():
            citations=pair.get(side+'_citations')
            if not isinstance(citations,list):raise ValueError('Identity citation list missing')
            for citation in citations:
                if not isinstance(citation,dict) or citation.get('evidence_id') not in record['labels']:
                    raise ValueError('Unknown cross-pair or cross-side evidence label')
                citation['evidence_id']=record['labels'][citation['evidence_id']]
    return result
