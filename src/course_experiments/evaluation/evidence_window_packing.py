"""问题相关的连续原文窗口打包；不读取参考答案、不发起模型请求。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata


POLICY = dict(version='question_sentence_budget_packing_v1',
              selection='first_top_k_unique_paragraphs_in_input_order',
              allocation='multiple_choice_serialized_byte_budget',
              source_text='one_exact_contiguous_window_per_original_paragraph',
              scoring='question_term_coverage_and_conditional_sentence_completeness',
              reference_answers_used=False, model_calls=0)

_SENTENCE_END = re.compile(r'[。！？!?；;]+[”’\"\']*|\.(?=\s|$)|\n+')
_CONDITION = re.compile(r'如果|只有|除非|前提|条件|若|当|\b(?:if|when|unless|provided)\b', re.I)
_CONDITIONAL_QUESTION = re.compile(r'何时|什么时候|条件|前提|何种|在什么|\b(?:when|condition|if|unless)\b', re.I)
_DEFINITION = re.compile(r'是指|定义为|称为|指的是|\b(?:defined as|refers to)\b', re.I)


def _bytes(value):
    return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))


def _terms(text):
    normalized = unicodedata.normalize('NFKC', text).casefold()
    result = set()
    for token in re.findall(r'[\u3400-\u9fff]+|[a-z0-9_]+', normalized):
        result.update([token] if token.isascii() or len(token) == 1 else
                      (token[i:i+2] for i in range(len(token)-1)))
    return result


def _sentences(text):
    ends = [match.end() for match in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    result = []; start = 0
    for end in ends:
        left, right = start, end
        while left < right and text[left].isspace(): left += 1
        while right > left and text[right-1].isspace(): right -= 1
        if left < right: result.append((left, right))
        start = end
    return result


@dataclass(frozen=True)
class _Option:
    group: int
    row: dict
    paragraph: tuple[str, str]
    start: int
    end: int
    complete_sentences: bool
    utility: int
    cost: int

    def context(self):
        return dict(book_id=self.row['book_id'], chunk_id=self.row['chunk_id'],
                    evidence_ids=list(self.row['evidence_ids']),
                    text=self.row['text'][self.start:self.end])


def _validated_groups(rows, top_k):
    groups = {}; seen = set()
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Context rows must be mappings')
        if any(not isinstance(row.get(k), str) or not row[k] for k in ['book_id', 'chunk_id', 'text']):
            raise ValueError('Source book, chunk and nonempty exact text are required')
        ids = row.get('evidence_ids')
        units = row.get('evidence')
        if (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]
                or not isinstance(units, list) or len(units) != 1 or not isinstance(units[0], dict)):
            raise ValueError('Each row requires one exact source evidence unit')
        unit = units[0]
        if (unit.get('id') != ids[0] or unit.get('text') != row['text']
                or unit.get('book_id', row['book_id']) != row['book_id']):
            raise ValueError('Evidence identity, textbook or exact source text differs')
        identity = (row['book_id'], ids[0])
        if identity in seen: raise ValueError('Duplicate source evidence identity within a book')
        seen.add(identity)
        source = row.get('original_source_id', ids[0])
        if not isinstance(source, str) or not source:
            raise ValueError('Original paragraph source identity must be a nonempty string')
        paragraph = (row['book_id'], source)
        if paragraph not in groups and len(groups) >= top_k: continue
        groups.setdefault(paragraph, []).append(row)
    return list(groups.items())


def _utility(text, query_terms, question, complete, group):
    overlap = len(query_terms & _terms(text))
    score = 500 - group + 1000 * overlap
    if overlap and _CONDITIONAL_QUESTION.search(question) and _CONDITION.search(text):
        score += 300
    if overlap and _DEFINITION.search(text): score += 100
    if complete: score += 40
    return score


def _options(paragraph, rows, group, question, query_terms, byte_budget, top_k):
    options = []
    for row in rows:
        text = row['text']; sentences = _sentences(text)
        template = dict(book_id=row['book_id'], chunk_id=row['chunk_id'],
                        evidence_ids=list(row['evidence_ids']), text='')
        if _bytes(template) + 2 >= byte_budget: continue
        anchors = sorted(range(len(sentences)), key=lambda i: (
            -len(query_terms & _terms(text[sentences[i][0]:sentences[i][1]])),
            -bool(_CONDITION.search(text[sentences[i][0]:sentences[i][1]])), i))[:8]
        spans = set()
        for anchor in anchors:
            for width in [1, 2, 3]:
                for first in range(max(0, anchor-width+1), anchor+1):
                    last = first+width-1
                    if last < len(sentences): spans.add((sentences[first][0], sentences[last][1], True))
        # 无完整句可放入时，从问题词附近保留连续字符，仍以精确JSON字节计费。
        for anchor in anchors:
            start, end = sentences[anchor]
            matches = [m.start() for term in sorted(query_terms)
                       for m in re.finditer(re.escape(term), text[start:end], flags=re.I)]
            point = start + (min(matches) if matches else 0)
            for divisor in range(1, top_k+1):
                cap = byte_budget // divisor
                left = max(start, point - max(1, cap//12))
                template['text'] = text[left:point+1]
                if _bytes(template)+2 > cap:
                    left = point
                    template['text'] = text[left:point+1]
                if _bytes(template)+2 > cap: continue
                lo, hi = point+1, end
                while lo < hi:
                    mid = (lo+hi+1)//2; template['text'] = text[left:mid]
                    if _bytes(template)+2 <= cap: lo = mid
                    else: hi = mid-1
                if lo > left: spans.add((left, lo, left == start and lo == end))
        for start, end, complete in sorted(spans):
            template['text'] = text[start:end]
            cost = _bytes(template)+2
            if cost <= byte_budget:
                options.append(_Option(group, row, paragraph, start, end, complete,
                                       _utility(template['text'], query_terms, question, complete, group), cost))
    # 同组被更便宜且得分不低的窗口支配时，不必进入预算求解。
    options.sort(key=lambda x: (x.cost, -x.utility, x.row['evidence_ids'][0], x.start, x.end))
    frontier = []; best = -1
    for option in options:
        if option.utility > best:
            frontier.append(option); best = option.utility
    return frontier


def pack_evidence_windows(rows, question, *, byte_budget=3000, top_k=5):
    """返回旧生成器上下文及可回查原文窗口；不改变原始来源或段落排序。"""
    if (not isinstance(question, str) or type(byte_budget) is not int or byte_budget < 2
            or type(top_k) is not int or not 1 <= top_k <= 5
            or not isinstance(rows, (list, tuple))):
        raise ValueError('Question text, UTF8 budget and top_k between one and five are required')
    groups = _validated_groups(rows, top_k)
    query_terms = _terms(question)
    choices = [_options(paragraph, records, group, question, query_terms, byte_budget, top_k)
               for group, (paragraph, records) in enumerate(groups)]
    # 非空JSON列表字节数恰等于所有元素的(序列化字节+2)之和。
    states = {0: (0, ())}
    for group_options in choices:
        updated = dict(states)
        for used, (score, kept) in states.items():
            for option in group_options:
                total = used + option.cost
                if total > byte_budget: continue
                proposed = (score + option.utility, kept + (option,))
                if total not in updated or proposed[0] > updated[total][0]:
                    updated[total] = proposed
        states = {}; best = -1
        for cost, value in sorted(updated.items()):
            if value[0] > best:
                states[cost] = value; best = value[0]
    _, selected = max(states.values(), key=lambda x: x[0])
    context = [option.context() for option in selected]
    selected_groups = {option.group for option in selected}
    spans = []
    for option, item in zip(selected, context):
        spans.append(dict(book_id=item['book_id'], chunk_id=item['chunk_id'],
            evidence_id=item['evidence_ids'][0], original_source_id=option.paragraph[1],
            char_start=option.start, char_end=option.end,
            source_sha256=hashlib.sha256(option.row['text'].encode('utf-8')).hexdigest(),
            window_sha256=hashlib.sha256(item['text'].encode('utf-8')).hexdigest(),
            complete_sentences=option.complete_sentences, question_term_utility=option.utility))
    omitted = []
    for i, (paragraph, records) in enumerate(groups):
        if i not in selected_groups:
            metadata_exceeds = all(_bytes(dict(book_id=r['book_id'], chunk_id=r['chunk_id'],
                evidence_ids=r['evidence_ids'], text=''))+2 >= byte_budget for r in records)
            omitted.append(dict(book_id=paragraph[0], original_source_id=paragraph[1],
                reason='metadata_exceeds_budget' if metadata_exceeds else 'shared_context_budget'))
    trace = dict(policy=POLICY, byte_budget=byte_budget, top_k=top_k,
                 context_bytes=_bytes(context), input_rows=len(rows), input_paragraphs=len(groups),
                 selected_count=len(context), source_spans=spans, omitted_paragraphs=omitted,
                 paragraph_order_preserved=True, source_text_rewritten=False,
                 reference_answers_used=False, model_calls=0)
    assert trace['context_bytes'] <= byte_budget
    return context, trace
