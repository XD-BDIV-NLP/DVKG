"""开发精排的数据边界与真实推理回执核验，不向评分器提供答案。"""
from __future__ import annotations

import hashlib
import math

from ..manifest import digest
from .evidence_window_packing import pack_evidence_windows
from .query_evidence_rerank import paragraph_key


MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = 'e099d4b9cdbd291b1569d416f19aaa6523570bc3'


def build_score_packet(tasks):
    packets = []; seen = set()
    for task in tasks:
        tid = task.get('task_id'); question = task.get('question')
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError('Unique nonempty task identities are required')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('A nonempty question is required')
        seen.add(tid); rows = task['candidates']
        if not rows or len(rows) > 64:
            raise ValueError('Each task requires one to 64 original paragraph candidates')
        keys = set(); candidates = []
        for row in rows:
            key = paragraph_key(row)
            if key in keys: raise ValueError('Repeated original paragraph candidate')
            keys.add(key)
            context, trace = pack_evidence_windows([row], question, byte_budget=1800, top_k=1)
            if len(context) != 1: raise ValueError('Candidate window could not be packed')
            span = trace['source_spans'][0]; text = context[0]['text']
            candidates.append(dict(key=list(key), text=text,
                char_start=span['char_start'], char_end=span['char_end'],
                input_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                source_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest()))
        packets.append(dict(task_id=tid, question=question, candidates=candidates))
    if not packets: raise ValueError('At least one development task is required')
    return dict(version='question_window_score_packet_v1', answers_included=False,
                model=MODEL, revision=REVISION, max_length=512, packets=packets)


def verify_score_receipt(packet, receipt):
    if (receipt.get('input_digest') != digest(packet)
            or receipt.get('model') != packet['model']
            or receipt.get('revision') != packet['revision']
            or receipt.get('max_length') != packet['max_length']
            or receipt.get('actual_inference') is not True):
        raise ValueError('Inference receipt does not match frozen input and model')
    expected = {p['task_id']:p for p in packet['packets']}; result = {}
    for item in receipt.get('items', []):
        tid = item.get('task_id')
        if tid not in expected or tid in result: raise ValueError('Unknown or repeated task receipt')
        sources = {tuple(c['key']):c for c in expected[tid]['candidates']}; scores = {}
        for scored in item.get('scores', []):
            key = tuple(scored.get('key', [])); value = scored.get('score')
            if (key not in sources or key in scores
                    or scored.get('input_sha256') != sources[key]['input_sha256']
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError('Missing, repeated or changed candidate score')
            scores[key] = float(value)
        if set(scores) != set(sources): raise ValueError('Inference receipt is incomplete')
        result[tid] = scores
    if set(result) != set(expected): raise ValueError('Inference task receipt is incomplete')
    return result
