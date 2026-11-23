from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from .manifest import digest

SOURCE_STATES = ('no_textbook_support', 'textbook_supported', 'revised_or_conflicting')


def read_contribution(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text('utf-8-sig'))
    if value.get('artifact_id') != digest({k: v for k, v in value.items() if k != 'artifact_id'}):
        raise ValueError(f'Artifact digest mismatch: {path}')
    return value


def write_immutable(path: str | Path, value: dict) -> dict:
    path = Path(path)
    if path.exists():
        old = (read_contribution(path) if 'artifact_id' in value
               else json.loads(path.read_text('utf-8-sig')))
        if old != value:
            raise ValueError(f'Immutable artifact changed: {path}')
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # 硬链接原子发布完整字节，目标已存在时绝不覆盖。
        try:
            os.link(temporary, path)
        except FileExistsError:
            return write_immutable(path, value)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def export_contribution(snapshot: dict, course_id: str, book_id: str | None,
                        scope: str, input_id: str) -> dict:
    graph = copy.deepcopy(snapshot)
    prefix = f'{course_id}/{book_id or "g0"}'
    maps = {table: {native: f'{prefix}/{singular}/{native}' for native in graph[table]}
            for table, singular in (('nodes','node'),('claims','claim'),('evidence','evidence'),('books','book'))}
    if book_id is None and (graph['books'] or graph['evidence']):
        raise ValueError('G0 contains textbook sources')
    if book_id is not None and set(graph['books']) != {book_id}:
        raise ValueError('Contribution must contain exactly its book')
    supports = []
    for table in maps:
        for native, row in graph[table].items():
            row['native_id'] = native
            row['id'] = maps[table][native]
            if 'book_id' in row:
                if row['book_id'] != book_id: raise ValueError('Foreign book evidence')
                row['native_book_id'] = row['book_id']
                row['book_id'] = maps['books'][book_id]
            for key in ('evidence_ids','definition_evidence_ids','mention_evidence_ids'):
                if key in row:
                    if not set(row[key]) <= set(maps['evidence']): raise ValueError('Unknown evidence')
                    row[key] = [maps['evidence'][eid] for eid in row[key]]
            if table == 'nodes':
                for definition in row.get('definitions', []):
                    ids = definition.get('evidence_ids', [])
                    if not set(ids) <= set(maps['evidence']): raise ValueError('Unknown definition evidence')
                    definition['evidence_ids'] = [maps['evidence'][eid] for eid in ids]
                if 'merged_from' in row:
                    row['native_merged_from'] = list(row['merged_from'])
                    row['merged_from'] = [f'{prefix}/node/{nid}' for nid in row['merged_from']]
            if table in ('nodes','claims'):
                if book_id is not None and not row.get('evidence_ids'):
                    raise ValueError('Textbook facts require evidence')
                if book_id is None and (row.get('evidence_ids') or row.get('definition_evidence_ids')):
                    raise ValueError('G0 must have no evidence')
                row['source_state'] = ('no_textbook_support' if book_id is None else
                    'revised_or_conflicting' if row.get('revision',1)>1 or row.get('status')!='active' and table=='claims' else
                    'textbook_supported')
            if table == 'claims':
                if 'merged_into' in row:
                    if row['merged_into'] not in maps['claims']: raise ValueError('Unknown merged claim')
                    row['merged_into'] = maps['claims'][row['merged_into']]
                for key in ('head','tail','node_id'):
                    if key in row['content']:
                        row['content'][key] = maps['nodes'][row['content'][key]]
                supports.append(dict(claim_id=row['id'], native_claim_id=native,
                    book_id=book_id, evidence_ids=row['evidence_ids'], source_state=row['source_state']))
        graph[table] = {maps[table][native]:row for native,row in graph[table].items()}
    # 事件保留原生审计内容；消费方使用显式映射解析其中的历史 ID。
    result = dict(format_version=1, course_id=course_id, book_id=book_id, scope=scope,
                  input_id=input_id, native_id_map=maps, graph=graph, supports=supports)
    result['artifact_id'] = digest(result)
    return result
