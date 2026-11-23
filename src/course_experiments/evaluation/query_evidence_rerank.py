"""问题条件驱动的离线精排与外部成对评分适配，不发起模型请求。"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from numbers import Real
import re
import unicodedata


POLICY = dict(version='query_evidence_rerank_v1', candidate_limit=64,
              offline_weights=dict(term_coverage=.65, condition_coverage=.20,
                                   intent_coverage=.10, route_prior=.05),
              pair_auxiliary_weights=dict(term_coverage=.02, condition_coverage=.01,
                                          intent_coverage=.005, route_prior=.005),
              redundancy_weight=.05, pair_redundancy_weight=.01,
              score_key='(book_id, original_source_id_or_unique_evidence_id)',
              source_text_unchanged=True, verified_support_claimed=False,
              tuning='fixed_synthetic_cases_only_not_selected_by_test_answer_scores')


def _normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def _terms(text):
    found = set()
    for part in re.findall(r'[\u3400-\u9fff]+|[a-z0-9_]+', _normalize(text)):
        if part.isascii() or len(part) == 1:
            found.add(part)
        else:
            found.update(part[i:i + 2] for i in range(len(part) - 1))
    return found


def paragraph_key(row):
    """同名原段落在不同教材中仍是两个独立候选。"""
    if not isinstance(row, Mapping):
        raise ValueError('Candidate row must be a mapping')
    book = row.get('book_id')
    ids = row.get('evidence_ids')
    if (not isinstance(book, str) or not book or not isinstance(ids, list)
            or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]
            or not isinstance(row.get('chunk_id'), str) or not row['chunk_id']
            or not isinstance(row.get('text'), str) or not row['text'].strip()):
        raise ValueError('Candidate requires one source evidence ID and its exact text/book/chunk')
    source = row.get('original_source_id')
    if source is None:
        source = row.get('source_evidence_id', ids[0])
    if not isinstance(source, str) or not source:
        raise ValueError('Original source identity must be a nonempty string')
    evidence = row.get('evidence')
    if (not isinstance(evidence, list) or len(evidence) != 1
            or not isinstance(evidence[0], Mapping)):
        raise ValueError('Candidate requires one exact source evidence unit')
    unit = evidence[0]
    if (unit.get('id') != ids[0] or unit.get('text') != row['text']
            or unit.get('book_id', book) != book):
        raise ValueError('Candidate evidence identity, textbook or exact text differs')
    return book, source


def _score_map(scores, keys, name, complete):
    if scores is None:
        return {}
    if not isinstance(scores, Mapping) or not set(scores) <= keys:
        raise ValueError(name + ' must use declared candidate paragraph keys')
    if complete and set(scores) != keys:
        raise ValueError(name + ' must cover every unique candidate paragraph')
    result = {}
    for key, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(name + ' contains a nonfinite or nonnumeric score')
        result[key] = float(value)
    return result


def _query_features(question):
    normalized = _normalize(question)
    conditions = set()
    for phrase in re.findall(r'(?:仅当|只有|除非|如果|若|当|在)[^，。？！?;；]{1,32}?(?:时|才|则)', normalized):
        conditions.update(_terms(phrase))
    for phrase in re.findall(r'(?:至少|至多|不超过|不少于|大于|小于|不大于|不小于|等于|不等于|>=|<=|>|<|=)?\s*-?\d+(?:\.\d+)?\s*(?:ms|kb|mb|gb|位|字节|次|个|秒|毫秒|%)?', normalized):
        conditions.add(re.sub(r'\s+', '', phrase))
    # 限定词保留原问题含义，未使用任何参考答案或支持标注。
    for word in ['必须', '仅', '只有', '除非', '不能', '不得', '不超过', '至少', '至多',
                 'only', 'unless', 'must', 'not']:
        if word in normalized:
            conditions.add(word)
    intents = []
    for query_terms, evidence_terms in [
        (['是什么', '定义', '含义', 'what is', 'define', 'meaning'], ['是', '指', '定义', '称为', 'means', 'defined', 'refers to']),
        (['为什么', '为何', '原因', 'why', 'cause'], ['因为', '由于', '因此', '导致', '所以', '原因', 'because', 'therefore']),
        (['区别', '异同', '比较', '相比', 'difference', 'compare'], ['而', '分别', '不同', '相同', '相比', '区别', 'whereas', 'unlike', 'difference']),
    ]:
        if any(t in normalized for t in query_terms):
            intents.append(evidence_terms)
    return _terms(question), conditions, intents


def rerank_evidence(question, rows, *, route_scores=None, pair_scores=None, limit=64):
    """返回原文候选及排序轨迹；外部评分键为paragraph_key(row)的二元组。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Nonempty question text is required')
    if type(limit) is not int or limit <= 0:
        raise ValueError('Candidate output limit must be a positive integer')
    rows = list(rows)
    keyed = [(paragraph_key(row), row) for row in rows]
    keys = {key for key, _ in keyed}
    pairs = _score_map(pair_scores, keys, 'pair_scores', complete=True)
    routes = _score_map(route_scores, keys, 'route_scores', complete=False)
    maximum_route = max([0., *routes.values()])
    query_terms, conditions, intents = _query_features(question)
    use_pairs = pair_scores is not None
    weights = POLICY['pair_auxiliary_weights' if use_pairs else 'offline_weights']
    representatives = {}
    evidence_seen = {}
    for key, row in keyed:
        source_id = key[0], row['evidence_ids'][0]
        source_identity = key, row['text']
        if source_id in evidence_seen and evidence_seen[source_id] != source_identity:
            raise ValueError('A frozen evidence ID has conflicting source identity or text')
        evidence_seen[source_id] = source_identity
        text = _normalize(row['text'])
        terms = _terms(row['text'])
        coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.
        compact = re.sub(r'\s+', '', text)
        condition = sum(c in terms or c in compact for c in conditions) / len(conditions) if conditions else 0.
        intent = sum(any(word in text for word in words) for words in intents) / len(intents) if intents else 0.
        route = max(0., routes.get(key, 0.)) / maximum_route if maximum_route > 0 else 0.
        components = dict(term_coverage=coverage, condition_coverage=condition,
                          intent_coverage=intent, route_prior=route,
                          raw_route_score=routes.get(key, 0.), pair_score=pairs.get(key))
        base = sum(weights[field] * components[field] for field in weights)
        if use_pairs:
            base += pairs[key]
        item = dict(key=key, row=row, terms=terms, base_score=base, components=components)
        previous = representatives.get(key)
        tie = lambda x: (-x['base_score'], x['row']['evidence_ids'][0], x['row']['chunk_id'])
        if previous is None or tie(item) < tie(previous):
            representatives[key] = item
    remaining = dict(representatives)
    ranked = []
    audit = []
    redundancy_weight = POLICY['pair_redundancy_weight' if use_pairs else 'redundancy_weight']
    while remaining:
        options = []
        for key, item in remaining.items():
            similarities = [len(item['terms'] & old['terms']) / len(item['terms'] | old['terms'])
                            if item['terms'] | old['terms'] else 0. for old in ranked]
            penalty = redundancy_weight * max(similarities, default=0.)
            score = item['base_score'] - penalty
            options.append((-score, key, item, penalty))
        negative, key, chosen, penalty = min(options, key=lambda x: (x[0], x[1]))
        ranked.append(chosen)
        components = dict(chosen['components'], redundancy_penalty=penalty,
                          weighted_terms=weights['term_coverage'] * chosen['components']['term_coverage'],
                          weighted_conditions=weights['condition_coverage'] * chosen['components']['condition_coverage'],
                          weighted_intent=weights['intent_coverage'] * chosen['components']['intent_coverage'],
                          weighted_route=weights['route_prior'] * chosen['components']['route_prior'])
        audit.append(dict(candidate_id=list(key), evidence_id=chosen['row']['evidence_ids'][0],
                          rank=len(ranked), selected=len(ranked) <= limit,
                          score=-negative, base_score=chosen['base_score'], components=components,
                          source_text_sha256=hashlib.sha256(chosen['row']['text'].encode('utf-8')).hexdigest()))
        del remaining[key]
    return [x['row'] for x in ranked[:limit]], dict(
        policy=deepcopy(POLICY), mode='external_pair_score_rerank' if use_pairs else 'offline_question_feature_rerank',
        input_rows=len(rows), unique_paragraphs=len(keys), returned_rows=min(limit, len(keys)),
        candidates=audit, pair_scores_provided=use_pairs, neural_inference_performed=False,
        pair_score_provenance='caller_supplied_unverified' if use_pairs else None,
        soft_route_prior=True, verified_support_claimed=False,
        question_features=dict(terms=sorted(query_terms), conditions=sorted(conditions),
                               intent_count=len(intents)))
