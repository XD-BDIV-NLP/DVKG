"""仅准备独立来源事实候选；语义判定和新拆分冻结前不得称为reference完成。"""
from __future__ import annotations

from collections import defaultdict

from ..manifest import digest
from .reference import source_units,curate_batch


POLICY = dict(version='source_only_atomic_reference_v2_prepare',seed=42,
    selection='book_and_source_chapter_round_robin_then_frozen_hash_order_all_reserves',
    independence='no_method_graph_scores_mapping_coverage_or_rank_input',
    target='400_unique_atomic_equivalence_groups_per_course_primary_AB_CD_supplement_separate',
    audit='machine_assisted_not_human_audited',activation='deferred_until_scope_identity_measurement_review')


def prepare_plan(course_id, books, corpora, schema):
    by_alias = {b['alias']:b['book_id'] for b in books}
    if len(by_alias)!=len(books) or not {'A','B'} <= set(by_alias) or set(by_alias)-set('ABCD'):
        raise ValueError('Explicit unique A/B source books required; C/D are separate supplements')
    primary = {by_alias[a]:200 for a in 'AB'}
    supplement = {by_alias[a]:100 for a in 'CD' if a in by_alias}
    units = []
    for book in sorted(by_alias.values()):
        corpus = corpora[book]
        chapters = {c['id']:str(c.get('chapter_id','unclassified')) for c in corpus['chunks']}
        buckets = defaultdict(list)
        for unit in source_units([corpus]):
            if unit['book_id']!=book:
                raise ValueError('Foreign source book in reference preparation')
            row = dict(unit_id=digest(unit),book_id=book,chunk_id=unit['chunk_id'],
                chapter_id=chapters[unit['chunk_id']],evidence_ids=[e['id'] for e in unit['evidence']])
            buckets[row['chapter_id']].append(row)
        order = sorted(buckets,key=lambda c:digest([42,course_id,book,c]))
        for chapter in order:
            buckets[chapter].sort(key=lambda u:digest([42,u['unit_id']]))
        for index in range(max(map(len,buckets.values()),default=0)):
            for chapter in order:
                if index<len(buckets[chapter]):
                    units.append(buckets[chapter][index])
    payload = dict(course_id=course_id,policy=POLICY,schema_id=digest(schema),target_facts=400,
        primary_book_targets=primary,supplement_book_targets=supplement,units=units,
        source_fingerprints={b:corpora[b]['fingerprint'] for b in sorted(by_alias.values())},
        status='source_only_prepared_no_model_calls_no_reference_claim',human_audited=False)
    return dict(payload,plan_id=digest(payload))


def candidate_payload(plan, unit_id, corpora, schema):
    if plan['plan_id']!=digest({k:v for k,v in plan.items() if k!='plan_id'}) or digest(schema)!=plan['schema_id']:
        raise ValueError('Source-only plan/schema changed')
    record = next(u for u in plan['units'] if u['unit_id']==unit_id)
    unit = next(u for u in source_units([corpora[record['book_id']]]) if u['chunk_id']==record['chunk_id'])
    if digest(unit)!=unit_id:
        raise ValueError('Source unit changed after candidate planning')
    return dict(unit=unit,schema=schema,course_id=plan['course_id'],requested_types=[],
        output_limits=dict(entities=32,facts=16,questions=0),
        instruction='Only source-supported atomic facts; retain direction, necessary scope and exact conditions; no padding; no graph inputs.')


def curate_candidate(packet, unit, schema):
    selected,audit = curate_batch(packet,unit,schema)
    return selected,dict(audit,human_audited=False,audit_status='machine_assisted_source_validated_candidates',
        semantic_truth_status='pending_independent_machine_judgment',reference_frozen=False)
