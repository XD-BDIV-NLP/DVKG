"""独立评分候选、前五段及实际上下文；金标不参与检索或打包。"""
from __future__ import annotations

import json


def _paragraph(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(x, str) or not x for x in value)):
        raise ValueError('Paragraph identity requires nonempty textbook and original source IDs')
    return tuple(value)


def _rows(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError('Source rows must be a sequence')
    index = {}
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Source row must be a mapping')
        paragraph = _paragraph((row.get('book_id'), row.get('original_source_id')))
        ids = row.get('evidence_ids'); units = row.get('evidence')
        if (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]
                or not isinstance(row.get('chunk_id'), str) or not row['chunk_id']
                or not isinstance(row.get('text'), str) or not row['text']
                or not isinstance(units, list) or len(units) != 1 or not isinstance(units[0], dict)):
            raise ValueError('Source row requires one exact evidence unit with chunk and text')
        unit = units[0]
        if (unit.get('id') != ids[0] or unit.get('text') != row['text']
                or unit.get('book_id', paragraph[0]) != paragraph[0]):
            raise ValueError('Source evidence identity, textbook or text differs')
        key = (paragraph[0], ids[0])
        binding = (paragraph, row['chunk_id'], row['text'])
        if key in index and index[key] != binding:
            raise ValueError('Conflicting source evidence identity')
        index[key] = binding
    return index


def score_stages(gold_paragraphs, candidates, selected, context, *, gold_quotes=None):
    """按原教材段落身份评分，精确检查实际上下文来源及字节预算。"""
    if not isinstance(gold_paragraphs, (list, tuple)) or not gold_paragraphs:
        raise ValueError('Nonempty original gold paragraph identities are required')
    gold = {_paragraph(value) for value in gold_paragraphs}
    candidate_index = _rows(candidates); selected_index = _rows(selected)
    if any(key not in candidate_index or candidate_index[key] != value
           for key, value in selected_index.items()):
        raise ValueError('Selected source must preserve a declared candidate identity and text')
    candidate_paragraphs = {value[0] for value in candidate_index.values()}
    selected_paragraphs = {value[0] for value in selected_index.values()}
    if len(selected_paragraphs) > 5:
        raise ValueError('Recall@5 requires at most five selected original paragraphs')
    if not isinstance(context, (list, tuple)) or len(context) > 5:
        raise ValueError('Context must contain at most five source windows')
    packed = {}; required = {'book_id', 'chunk_id', 'evidence_ids', 'text'}
    for item in context:
        if (not isinstance(item, dict) or set(item) != required
                or not isinstance(item['book_id'], str) or not item['book_id']
                or not isinstance(item['chunk_id'], str) or not item['chunk_id']
                or not isinstance(item['evidence_ids'], list) or len(item['evidence_ids']) != 1
                or not isinstance(item['evidence_ids'][0], str)
                or not isinstance(item['text'], str) or not item['text'].strip()):
            raise ValueError('Context requires the four exact source-window fields')
        key = (item['book_id'], item['evidence_ids'][0])
        if key not in selected_index:
            raise ValueError('Context evidence does not belong to selected sources')
        paragraph, chunk, original = selected_index[key]
        if chunk != item['chunk_id'] or item['text'] not in original:
            raise ValueError('Context must preserve chunk identity and one contiguous original substring')
        packed.setdefault(paragraph, []).append(item['text'])
    context_bytes = len(json.dumps(context, ensure_ascii=False).encode('utf-8'))
    if context_bytes > 3000:
        raise ValueError('Context exceeds the common 3000 UTF8 byte budget')
    quote_coverage = None
    if gold_quotes is not None:
        if not isinstance(gold_quotes, (list, tuple)):
            raise ValueError('Gold quotes must be original textbook/source/quotation triples')
        quotes = set()
        for value in gold_quotes:
            if (not isinstance(value, (tuple, list)) or len(value) != 3
                    or not isinstance(value[2], str) or not value[2].strip()):
                raise ValueError('Gold quote requires a nonempty exact quotation')
            paragraph = _paragraph(value[:2])
            if paragraph not in gold: raise ValueError('Gold quote does not belong to a gold paragraph')
            quotes.add((paragraph, value[2]))
        if quotes:
            covered = sum(any(quote in text for text in packed.get(paragraph, []))
                          for paragraph, quote in quotes)
            quote_coverage = covered / len(quotes)
    denominator = len(gold)
    return dict(candidate_paragraph_recall=len(gold & candidate_paragraphs)/denominator,
                paragraph_recall_at5=len(gold & selected_paragraphs)/denominator,
                packed_paragraph_recall=len(gold & packed.keys())/denominator,
                context_bytes=context_bytes, gold_quote_coverage=quote_coverage)
