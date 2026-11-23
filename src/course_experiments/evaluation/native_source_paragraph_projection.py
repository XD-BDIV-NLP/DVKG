"""Deterministic, source-position-derived paragraph ranks; no query or gold input.

This defines a projected ranking. It does not recover a native paragraph
relevance score from a method that retrieves larger chunks.
"""
from __future__ import annotations

POLICY = dict(
    version='native_source_position_projected_paragraph_R5_v1',
    native_order='unchanged_saved_provided_context_order',
    within_source='relative_character_start_then_original_paragraph_order_then_canonical_id',
    canonical_id='book_id/paragraph/source.source_id',
    duplicates='first_occurrence_only', k=5,
    gold_or_question_used_for_ranking=False, added_semantic_reranking=False,
    native_paragraph_rank_claim=False,
    ambiguity='all_exact_proven_locations_retained;stable_source_position_tie_break;reported',
)


def project_sources(rows, book, evidence, passages, k=5):
    if k != 5:
        raise ValueError('This frozen metric requires k=5')
    seen = set()
    order, trace, row_counts = [], [], []
    duplicates = ambiguous = 0
    previous_rank = 0

    def original(eid, chunk):
        item = evidence.get(eid)
        if not item or item.get('book_id') != book or item.get('chunk_id') != chunk:
            raise ValueError('Evidence is outside the exact original book/source')
        source = item['source']
        if not isinstance(source.get('source_id'), str) or not source['source_id']:
            raise ValueError('Canonical source paragraph identifier missing')
        if type(source.get('paragraph_order')) is not int:
            raise ValueError('Original paragraph position missing')
        return item, book + '/paragraph/' + source['source_id']

    for ordinal, row in enumerate(rows, 1):
        if row.get('book_id') != book or not isinstance(row.get('text'), str) or not row['text']:
            raise ValueError('Nonempty exact target-book source required')
        if 'rank' in row:
            rank = row['rank']
            if type(rank) is not int or rank <= previous_rank:
                raise ValueError('Saved native ranks must increase in the supplied order')
            previous_rank = rank
        candidates = []
        if 'evidence_id' in row:
            eid = row['evidence_id']
            item, pid = original(eid, row.get('chunk_id'))
            start, end = row.get('evidence_start'), row.get('evidence_end')
            if (type(start) is not int or type(end) is not int or
                    not 0 <= start < end <= len(item['text']) or
                    item['text'][start:end] != row['text'] or
                    row.get('evidence_ids') != [eid]):
                raise ValueError('Evidence fragment is not its claimed original character span')
            candidates.append((0, item['source']['paragraph_order'], pid, eid, start, end))
        else:
            positions = row.get('positions')
            if not isinstance(positions, list) or not positions:
                raise ValueError('Native chunk has no exact original source positions')
            if len(positions) > 1:
                ambiguous += 1
            expected_ids = set()
            for position in positions:
                chunk = position.get('chunk_id')
                passage = passages.get(chunk)
                start, end = position.get('passage_start'), position.get('passage_end')
                if (position.get('book_id') != book or position.get('position_unit') != 'unicode_character'
                        or not passage or passage.get('book_id') != book
                        or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(passage['text'])
                        or passage['text'][start:end] != row['text']):
                    raise ValueError('Native chunk text does not equal its original source window')
                spans = [s for s in passage['evidence_spans']
                         if s['start'] < end and s['end'] > start]
                if position.get('evidence_spans') != spans:
                    raise ValueError('Native window evidence spans differ from the full original passage')
                for span in spans:
                    eid = span['evidence_id']
                    item, pid = original(eid, chunk)
                    if passage['text'][span['start']:span['end']] != item['text']:
                        raise ValueError('Passage evidence span differs from the original paragraph')
                    left, right = max(start, span['start']), min(end, span['end'])
                    expected_ids.add(eid)
                    candidates.append((left-start, item['source']['paragraph_order'], pid,
                                       eid, left-span['start'], right-span['start']))
            if set(row.get('evidence_ids', [])) != expected_ids:
                raise ValueError('Advertised evidence identifiers differ from the proven source windows')
        row_counts.append(len({candidate[2] for candidate in candidates}))
        for offset, paragraph_order, pid, eid, start, end in sorted(candidates):
            if pid in seen:
                duplicates += 1
                continue
            seen.add(pid)
            order.append(pid)
            trace.append(dict(projected_rank=len(order), source_row_ordinal=ordinal,
                              source_row_rank=row.get('rank'), relative_character_start=offset,
                              paragraph_order=paragraph_order, paragraph_id=pid,
                              evidence_id=eid, evidence_start=start, evidence_end=end))
    return dict(paragraph_ids=order[:k], unique_paragraphs_total=len(order),
                duplicate_paragraph_hits=duplicates, ambiguous_source_rows=ambiguous,
                returned_source_rows=len(rows), per_source_row_paragraph_counts=row_counts,
                projection_trace=trace)
