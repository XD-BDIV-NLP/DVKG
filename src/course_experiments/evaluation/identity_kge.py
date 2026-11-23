"""已完成机器身份资格到独立KGE拆分的薄适配；不调用模型或训练。"""
from __future__ import annotations

import copy

from ..manifest import digest
from ..splits import canonical,canonical_reference,freeze_split,project_graph,validate_split
from .scope_mapping_preflight import qualification


POLICY = dict(version='qualified_identity_kge_v1',seed=42,ratios=[.8,.1,.1],
    population='all_independent_reference_facts_no_coverage_or_rank_selection',
    reference_equivalence='independent_completed_source_only_reference_components_and_existing_explicit_aliases_no_graph_induced_union',
    fact_equivalence='existing_inverse_symmetric_and_conditional_plain_triple_groups',
    conservative_holdout='all_remaining_lexical_endpoint_risk_groups_same_fold_even_different_sense_or_insufficient_no_identity_merge',
    mapping='completed_source_and_prior_full_all_development_gates_no_failed_pairs_conflicts_rejected',
    prior='prior_generated_definition_identity_is_not_textbook_relation_support',
    graph='original_bytes_preserved_identity_projection_only_no_added_edges_or_types',
    coverage='full_fixed_test_and_validation_endpoint_and_relation_training_coverage',
    training='existing_evaluate_kg_quality_external_protocol_and_PyKEEN_unchanged',
    old_scores='new_identity_protocol_never_relabel_old_training_results',human_audited=False)
STAGES=('source32','source-full','prior24-single','prior24-batch','prior-full')


def qualify_mappings(stages,source_origins,prior_origins,preparation_id,*,reference_qualification=None):
    from .run_scope_identity_gate import identity_projection
    for name in STAGES:
        row=stages.get(name,{})
        decisions=row.get('decisions',{})
        if (row.get('stage')!=name or row.get('status')!='completed' or
                row.get('preparation_id')!=preparation_id or row.get('failed_pairs')!=0 or
                row.get('candidate_count')!=len(decisions) or not decisions or
                any(d.get('status')!='completed' or d.get('citation_valid') is not True or
                    d.get('eligible_identity_edge') and d.get('reversed_positive_confirmed') is not True
                    for d in decisions.values())):
            raise ValueError('Incomplete or invalid identity qualification stage: '+name)
    for name in ('source32','prior24-batch'):
        if stages[name].get('comparison',{}).get('passes_declared_gate') is not True:
            raise ValueError('Declared identity development gate did not pass: '+name)
    branches=[];all_origins=[];all_decisions={}
    for name,origins,basis in [('source-full',source_origins,'two_sided_textbook_source'),
                               ('prior-full',prior_origins,'prior_generated_definition')]:
        decisions=stages[name]['decisions']
        if {r['candidate_id'] for r in origins}!=set(decisions):
            raise ValueError('Full identity population differs from frozen origins')
        expected=identity_projection(origins,decisions)
        if stages[name].get('identity_projection')!=expected:
            raise ValueError('Recorded identity projection differs from validated decisions')
        if all_decisions.keys() & decisions.keys():raise ValueError('Source/prior candidate identities overlap')
        branches.extend((r,basis) for r in origins);all_origins.extend(origins);all_decisions.update(decisions)
    joint=identity_projection(all_origins,all_decisions)
    if 'combined_identity_projection' in stages['prior-full'] and stages['prior-full']['combined_identity_projection']!=joint:
        raise ValueError('Joint identity conflict projection differs from frozen full result')
    original_joint=joint
    if reference_qualification is not None:
        if reference_qualification.get('status')!='completed_reference_identity_qualification':
            raise ValueError('Independent source-only reference qualification incomplete')
        component_maps={}
        for course,components in reference_qualification['components'].items():
            mapping={}
            for component in components:
                ids=sorted(set(component))
                if len(ids)<2 or set(ids)&mapping.keys():raise ValueError('Independent reference components overlap or are malformed')
                mapping.update({key:ids[0] for key in ids})
            component_maps[course]=mapping
        normalized=[];branches=[];all_decisions=copy.deepcopy(all_decisions)
        for origins,basis in [(source_origins,'two_sided_textbook_source'),(prior_origins,'prior_generated_definition')]:
            for origin in origins:
                row=copy.deepcopy(origin);old=row['reference_entity_id']
                row.update(original_reference_entity_id=old,
                    reference_entity_id=component_maps.get(row['course_id'],{}).get(old,old))
                normalized.append(row);branches.append((row,basis))
                decision=all_decisions[row['candidate_id']]
                decision['eligible_identity_edge']=bool(decision.get('primary_record',{}).get('eligible_identity_edge') and
                    (decision.get('reverse') or {}).get('eligible_identity_edge') and not decision.get('veto'))
        joint=identity_projection(normalized,all_decisions)
    accepted={digest(r) for r in joint['mapping']}
    result=dict(policy=POLICY,preparation_id=preparation_id,usable_for_KGE=True,
        mapping=[dict(r,identity_basis=basis) for r,basis in branches if digest(r) in accepted],
        rejections=joint['rejections'],identity_projection_id=joint['identity_projection_id'],
        original_unchanged_joint_projection_id=original_joint['identity_projection_id'],
        reference_qualification_id=(reference_qualification or {}).get('qualification_id'),
        reference_components_inferred_from_graphs=False,original_graphs_modified=False,
        old_splits_modified=False,new_split_required=True,human_audited=False)
    result['qualification_id']=digest(result)
    return result


def freeze_reference(reference,qualification_id,*,reference_qualification=None):
    if not qualification_id:raise ValueError('Completed mapping qualification identity required')
    from .reference_identity import candidate_pairs
    candidate_count=len(candidate_pairs(canonical_reference(reference)['entities']))
    if candidate_count and reference_qualification is None:
        raise ValueError('Unresolved cross-scope reference identity candidates require independent source-only qualification')
    # 仅采用独立来源门禁，绝不从竞争图的覆盖或冲突自动归并reference。
    result=copy.deepcopy(reference)
    if reference_qualification is not None:
        if (reference_qualification.get('status')!='completed_reference_identity_qualification' or
                reference_qualification.get('original_reference_digests',{}).get(reference['course_id'])!=digest(reference)):
            raise ValueError('Independent reference qualification differs from this original reference')
        from .run_scope_identity_gate import freeze_identity_split
        result,_=freeze_identity_split(canonical_reference(reference),reference_qualification['components'].get(reference['course_id'],[]))
    result['identity_kge_protocol']=dict(policy=POLICY,qualification_id=qualification_id,
        original_reference_digest=digest(reference),all_original_fact_ids=[f['fact_id'] for f in reference['facts']],
        reference_qualification_id=(reference_qualification or {}).get('qualification_id'),
        independently_qualified_reference_pairs=candidate_count,fold_membership_may_match_prior_seed42=True)
    if 'reference_id' in result:result['reference_id']=digest({k:v for k,v in result.items() if k!='reference_id'})
    split=freeze_split(result)
    if reference_qualification is not None:split=cohold_lexical_fact_risks(result,split)
    if sorted(f['fact_id'] for f in reference['facts'])!=sorted(fid for row in split['groups'].values() for fid in row['fact_ids']):
        raise ValueError('Independent fact population changed during grouping')
    return result,split


def cohold_lexical_fact_risks(reference,split):
    from collections import defaultdict
    from .reference_identity_audit import audit_reference
    validate_split(split);result=copy.deepcopy(split)
    risks=audit_reference(reference)['potential_fact_group_risks'];parent={gid:gid for gid in split['groups']}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    for risk in risks:
        ids=risk['actual_group_ids']
        for gid in ids[1:]:
            a,b=find(ids[0]),find(gid);parent[max(a,b)]=min(a,b)
    members=defaultdict(list)
    for gid in parent:members[find(gid)].append(gid)
    blocks={}
    for ids in members.values():
        ids=sorted(ids);key=ids[0] if len(ids)==1 else digest(dict(policy='lexical_fact_cohold_v1',group_ids=ids))
        blocks[key]=ids
    ordered=sorted(blocks,key=lambda key:digest([42,key]));n=len(ordered);a=int(n*.8);b=int(n*.1)
    block_folds=dict(train=ordered[:a],validation=ordered[a:a+b],test=ordered[a+b:])
    result['folds']={key:[gid for bid in ids for gid in blocks[bid]] for key,ids in block_folds.items()}
    result['identity_holdout_blocks']=blocks;result['identity_holdout_block_folds']=block_folds
    result['identity_holdout_block_policy']=dict(version='conservative_lexical_fact_cohold_v1',seed=42,
        different_sense_handling='conservatively_coheld_same_as_insufficient',entity_identity_merge=False,
        candidate_source='all_remaining_same_name_explicit_alias_endpoint_inverse_condition_lexical_risks_no_graph_rank_coverage',
        ratios_applied_to='holdout_blocks',block_count=n,actual_group_count=len(split['groups']),
        multi_group_blocks=sum(len(ids)>1 for ids in blocks.values()),lexical_risk_clusters=len(risks),
        actual_group_counts={key:len(ids) for key,ids in result['folds'].items()},
        actual_group_ratios={key:len(ids)/len(split['groups']) if split['groups'] else 0 for key,ids in result['folds'].items()},
        all_reference_facts_preserved=True,actual_canonical_group_ids_unchanged=True)
    result['policy']+='; all remaining lexical endpoint fact-risk groups conservatively share seed42 holdout blocks'
    result['split_id']=digest({k:v for k,v in result.items() if k!='split_id'});validate_split(result)
    assignment={gid:key for key,ids in result['folds'].items() for gid in ids}
    if any(len({assignment[gid] for gid in risk['actual_group_ids']})!=1 for risk in risks):raise ValueError('Lexical reference risk block crosses folds')
    return result


def project_identity_graph(graph,reference,split,mapping,*,graph_sha,capabilities=None):
    status=qualification(graph,capabilities)
    if status['status']!='eligible_typed_graph':raise ValueError(status['status'])
    canonical_ref=canonical_reference(reference);refs={r['entity_id']:r for r in canonical_ref['entities']}
    alias_map=canonical_ref['reference_alias_mapping'];native={e['entity_id']:e for e in graph['entities']}
    if len(native)!=len(graph['entities']):raise ValueError('Duplicate native entity identity')
    mapped={};provenance={}
    for row in mapping:
        if row['graph_record']['sha256']!=graph_sha:raise ValueError('Graph SHA differs from frozen machine mapping')
        if row['course_id']!=reference['course_id']:raise ValueError('Cross-course identity projection forbidden')
        source=row['graph_entity_id'];target=alias_map.get(row['reference_entity_id'])
        if source not in native or target not in refs:raise ValueError('Unknown native/reference identity endpoint')
        if source in mapped and mapped[source]!=target:raise ValueError('Native entity has conflicting reference targets')
        mapped[source]=target;provenance.setdefault(source,[]).append(copy.deepcopy(row))
    # 仅为现有project_graph提供已资格通过的身份；边和原始类型保持不变。
    view=dict(entities=[dict(entity_id=source,canonical_name=refs[target]['canonical_name'],
        aliases=[target],scope=refs[target].get('scope',''),entity_type=native[source].get('entity_type',native[source].get('type')))
        for source,target in sorted(mapped.items())],relations=copy.deepcopy(graph['relations']))
    projection=project_graph(view,reference,split);projection['graph_id']=digest(graph)
    projection['audit'].update(identity_mapping_provenance=provenance,identity_projection_view_id=digest(view),
        original_graph_file_sha256=graph_sha,unmapped_native_entities=sorted(set(native)-set(mapped)))
    edge_sources=[];heldout=set(split['folds']['validation']+split['folds']['test'])
    for i,edge in enumerate(graph['relations']):
        h=mapped.get(edge.get('head_entity_id'));t=mapped.get(edge.get('tail_entity_id'));r=edge.get('relation')
        if h and t and r in projection['relation_vocabulary']+['consists_of']:
            triple=list(canonical(h,r,t))
            edge_sources.append(dict(original_relation_id=edge.get('relation_id',str(i)),triple=triple,
                heldout_removed=digest(triple) in heldout,
                identity_basis=sorted({p['identity_basis'] for endpoint in (edge['head_entity_id'],edge['tail_entity_id']) for p in provenance[endpoint]})))
    expected={tuple(r['triple']) for r in edge_sources if not r['heldout_removed']}
    if expected!={tuple(t) for t in projection['training']}:raise ValueError('Projection introduced or lost a mapped native fact')
    projection['audit']['identity_edge_sources']=edge_sources
    projection['identity_disclosure']=dict(policy=POLICY,prior_identity_is_textbook_support=False,
        original_graph_unchanged=True,reference_equivalence_from_graphs=False,
        identity_only_no_relation_support_inferred=True)
    trained={x for h,_,t in projection['training'] for x in (h,t)};relations={r for _,r,_ in projection['training']}
    coverage={}
    for fold in ('test','validation'):
        details=[dict(group_id=gid,head_in_training=t[0] in trained,tail_in_training=t[2] in trained,
            relation_in_training=t[1] in relations,eligible=eligible) for gid,t,eligible in
            zip(projection[fold+'_ids'],projection[fold],projection['covered_'+fold+'_mask'])]
        coverage.update({'fixed_'+fold+'_count':len(details),'eligible_'+fold+'_count':sum(r['eligible'] for r in details),fold:details})
    projection['coverage_audit']=coverage
    projection['projection_id']=digest({k:v for k,v in projection.items() if k!='projection_id'})
    return projection
