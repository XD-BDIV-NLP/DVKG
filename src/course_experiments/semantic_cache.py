from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from contextlib import ExitStack, contextmanager
from pathlib import Path

from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime, digest, dumps
from domain_ckg_workspace.ckg_incremental.semantic_merge import SemanticMerger
from .contributions import read_contribution
from .fusion import _validate
from .manifest import read_document

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ('domain_ckg_workspace/ckg_incremental/semantic_merge.py',
              'domain_ckg_workspace/ckg_incremental/runtime.py',
              'course_experiments/contributions.py', 'course_experiments/construction.py',
              'course_experiments/textbook.py')


def _checked_file(record):
    path = Path(record['path']).resolve(strict=True)
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != record['sha256']:
        raise ValueError(f'Frozen file hash mismatch: {path}')
    return path


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _no_sidecars(path):
    if any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('Source must be a closed immutable SQLite backup without sidecars')


@contextmanager
def _open_snapshot(record, runtime):
    path = Path(record['path']).resolve(strict=True)
    if path == (runtime.root / 'model_cache.sqlite').resolve() or path.name == 'model_cache.sqlite':
        raise ValueError('Live model cache cannot be a source snapshot')
    _no_sidecars(path)
    before = _stamp(path)
    _checked_file(record)
    if _stamp(path) != before:
        raise ValueError('Source snapshot changed during validation')
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        primary = [row[1] for row in db.execute('PRAGMA table_info(semantic_decisions)') if row[5]]
        if primary != ['key']:
            raise ValueError('Source cache requires the original unique primary key')
        plan = [row[3] for row in db.execute(
            'EXPLAIN QUERY PLAN SELECT value FROM semantic_decisions WHERE key=?', ('probe',))]
        if not plan or any('SEARCH' not in row or 'INDEX' not in row for row in plan):
            raise ValueError('Source semantic cache requires an indexed key lookup')
        yield db
    finally:
        db.close()


def _valid_decision(row, pair):
    if not isinstance(row, dict) or row.get('pair_id') != pair['pair_id']:
        return False
    if row.get('decision') not in ('merge', 'separate', 'unresolved'):
        return False
    if not isinstance(row.get('reason'), str) or not row['reason'].strip():
        return False
    for key in ('left_qualifier', 'right_qualifier'):
        if not isinstance(row.get(key, ''), str) or len(row.get(key, '')) > 60:
            return False
    ids = row.get('evidence_ids')
    supplied = {row['id'] for side in ('left', 'right') for row in pair[side]['evidence']}
    return (isinstance(ids, list) and all(isinstance(e, str) for e in ids)
            and set(ids) <= supplied)


class _Source:
    def __init__(self, descriptor, contribution, runtime):
        self.record = copy.deepcopy(descriptor)
        self.contribution = contribution
        self.runtime = runtime
        self.stack = ExitStack()

    def __enter__(self):
        try:
            self._validate()
            self.db = self.stack.enter_context(_open_snapshot(self.record['cache'], self.runtime))
            self.path = Path(self.record['cache']['path']).resolve()
            self.stamp = _stamp(self.path)
            self.query_plan = [r[3] for r in self.db.execute(
                'EXPLAIN QUERY PLAN SELECT value FROM semantic_decisions WHERE key=?', ('probe',))]
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _validate(self):
        record, value = self.record, self.contribution
        _validate(value)
        if record.get('source_kind') != 'completed_independent_full_book':
            raise ValueError('Only completed independent full-book sources are allowed')
        if value['scope'] != 'full' or not value['book_id']:
            raise ValueError('Source must be a full independent textbook contribution')
        for field in ('course_id', 'book_id', 'scope', 'input_id', 'artifact_id'):
            if record.get(field) != value[field]:
                raise ValueError(f'Source identity mismatch: {field}')
        frozen = read_contribution(_checked_file(record['contribution']))
        if frozen != value:
            raise ValueError('Current contribution differs from the frozen source')
        identity = read_document(_checked_file(record['construction_identity']))
        if identity != {k:value[k] for k in ('course_id', 'book_id', 'scope', 'input_id')}:
            raise ValueError('Source construction identity mismatch')
        manifest = read_document(_checked_file(record['construction_manifest']))
        if manifest.get('manifest_id') != digest({k:v for k,v in manifest.items() if k != 'manifest_id'}):
            raise ValueError('Source construction manifest digest mismatch')
        courses = [c for c in manifest['courses']['courses'] if c['course_id'] == value['course_id']]
        if len(courses) != 1 or value['book_id'] not in {b['book_id'] for b in courses[0]['books']}:
            raise ValueError('Source book is outside the frozen construction course')
        assets = manifest['files']['assets']
        for relative in CODE_PATHS:
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            matches = [r for r in assets.values() if Path(r['path']).as_posix().endswith('/' + relative)]
            if (record.get('code_hashes', {}).get(relative) != actual or len(matches) != 1
                    or matches[0]['sha256'] != actual):
                raise ValueError(f'Frozen algorithm code mismatch: {relative}')
            _checked_file(matches[0])
        config = read_document(_checked_file(assets['runtime_config']))
        templates = {}
        for name in {name for pair in ModelRuntime.FILES.values() for name in pair}:
            matches = list({(str(Path(r['path']).resolve()), r['sha256']): r
                            for r in assets.values() if Path(r['path']).name == name}.values())
            if len(matches) != 1:
                raise ValueError('Frozen prompt asset is missing or ambiguous')
            templates[name] = _checked_file(matches[0]).read_text('utf-8-sig')
        semantic = digest(dict(templates={k:templates[k] for k in ModelRuntime.FILES['semantic_identity']}, llm=config['llm']))
        runtime = digest(dict(templates={k:v for k,v in templates.items() if k not in ModelRuntime.FILES['semantic_identity']},
                              llm=config['llm'], embedding=config['embedding']))
        current_semantic = digest(dict(templates={k:self.runtime.templates[k] for k in ModelRuntime.FILES['semantic_identity']},
                                       llm=self.runtime.config['llm']))
        current_runtime = digest(dict(templates={k:v for k,v in self.runtime.templates.items() if k not in ModelRuntime.FILES['semantic_identity']},
                                      llm=self.runtime.config['llm'], embedding=self.runtime.config['embedding']))
        if (semantic != record['semantic_fingerprint'] or semantic != self.runtime.semantic_fingerprint
                or semantic != current_semantic):
            raise ValueError('Semantic model or prompt fingerprint mismatch')
        if (runtime != record['runtime_fingerprint'] or runtime != self.runtime.fingerprint
                or runtime != current_runtime):
            raise ValueError('Construction model or embedding fingerprint mismatch')
        corpus = read_document(_checked_file(manifest['files']['sources'][value['book_id']]))
        if corpus.get('fingerprint') != digest({k:v for k,v in corpus.items() if k != 'fingerprint'}):
            raise ValueError('Source corpus fingerprint mismatch')
        scoped = dict(corpus, construction_scope='full', source_corpus_fingerprint=corpus['fingerprint'], smoke_chunks=None)
        scoped.pop('fingerprint')
        if value['input_id'] != digest(dict(manifest=manifest['manifest_id'], book_id=value['book_id'], corpus=digest(scoped))):
            raise ValueError('Source input is not this full independent construction')
        self.maps = value['native_id_map']
        self.inverse = {table:{v:k for k,v in mapping.items()} for table,mapping in self.maps.items()}
        if any(len(self.inverse[t]) != len(mapping) for t,mapping in self.maps.items()):
            raise ValueError('Source namespaces are not bijective')
        if set(self.maps['books']) != {value['book_id']}:
            raise ValueError('Source contains prior or foreign book state')
        book_key = self.maps['books'][value['book_id']]
        if any(e.get('book_id') != book_key for e in value['graph']['evidence'].values()):
            raise ValueError('Source contains foreign evidence')
        if any(not n.get('evidence_ids') for n in value['graph']['nodes'].values()):
            raise ValueError('Source contains unsupported prior nodes')

    def _native_row(self, table, identifier):
        row = copy.deepcopy(self.contribution['graph'][table][identifier])
        row['id'] = self.inverse[table][identifier]
        if 'book_id' in row:
            row['book_id'] = self.inverse['books'][row['book_id']]
        for field in ('evidence_ids', 'definition_evidence_ids', 'mention_evidence_ids'):
            if field in row:
                row[field] = [self.inverse['evidence'][eid] for eid in row[field]]
        for definition in row.get('definitions', []):
            definition['evidence_ids'] = [self.inverse['evidence'][eid] for eid in definition.get('evidence_ids', [])]
        return row

    def rebind(self, pair, left_id, right_id):
        nodes = [self._native_row('nodes', nid) for nid in (left_id, right_id)]
        ids = {eid for node in nodes for field in ('evidence_ids', 'definition_evidence_ids') for eid in node.get(field, [])}
        ids.update(eid for node in nodes for definition in node.get('definitions', []) for eid in definition.get('evidence_ids', []))
        evidence = {eid:self._native_row('evidence', self.maps['evidence'][eid]) for eid in ids}
        books = {bid:self._native_row('books', nid) for bid,nid in self.maps['books'].items()}
        original = SemanticMerger(None, evidence, books)._pair(*nodes)
        rebound = copy.deepcopy(original)
        for side in ('left', 'right'):
            for definition in rebound[side]['definition_variants']:
                definition['evidence_ids'] = [self.maps['evidence'][eid] for eid in definition['evidence_ids']]
            for row in rebound[side]['evidence']:
                row['id'] = self.maps['evidence'][row['id']]
                if 'book_id' in row:
                    row['book_id'] = self.maps['books'][row['book_id']]
        rebound['pair_id'] = digest({k:v for k,v in rebound.items() if k != 'pair_id'})
        # 完整载荷比较保留顺序、方向及所有非 ID 字段。
        return original if dumps(rebound) == dumps(pair) else None

    def lookup(self, pair):
        _no_sidecars(self.path)
        if _stamp(self.path) != self.stamp:
            raise ValueError('Immutable source snapshot changed')
        key = digest([self.record['semantic_fingerprint'], pair['pair_id']])
        row = self.db.execute('SELECT value FROM semantic_decisions WHERE key=?', (key,)).fetchone()
        return key, row[0] if row else None


def prefill_exact_cache(g0, contributions, active_books, *, runtime, sources):
    _validate(g0)
    if g0['book_id'] is not None or g0['scope'] != 'full':
        raise ValueError('G0 must be a full prior contribution')
    books = {}
    for value in contributions:
        _validate(value)
        if value['course_id'] != g0['course_id'] or not value['book_id'] or value['book_id'] in books:
            raise ValueError('Duplicate, missing or foreign contribution identity')
        books[value['book_id']] = value
    active = sorted(set(active_books))
    if not set(active) <= books.keys() or any(books[b]['scope'] != 'full' for b in active):
        raise ValueError('Exact prefill requires selected full textbook contributions')
    report = dict(policy='exact_namespace_cache_prefill_v1',
                  counts={k:0 for k in ('attempted','hit','noneligible','mismatch','nohit','invalid','preservedexisting')},
                  hits=[], invalid=[], sources=[], semantic_provider_calls=0,
                  cost_accounting='Reused historical source decisions; no new semantic provider calls. Embeddings may use the normal runtime.',
                  actual_savings_verified=False)
    with ExitStack() as stack:
        by_book = {}
        for descriptor in sources:
            bid = descriptor['book_id']
            if bid not in active or bid in by_book:
                raise ValueError('Source must identify one selected book exactly once')
            source = stack.enter_context(_Source(descriptor, books[bid], runtime))
            by_book[bid] = source
            report['sources'].append(dict(descriptor=copy.deepcopy(descriptor), query_plan=source.query_plan))
        selected = [g0] + [books[b] for b in active]
        nodes = {k:v for c in selected for k,v in c['graph']['nodes'].items()}
        evidence = {k:v for c in selected for k,v in c['graph']['evidence'].items()}
        book_rows = {k:v for c in selected for k,v in c['graph']['books'].items()}
        claims = {k:v for c in selected for k,v in c['graph']['claims'].items()}
        owner = {nid:b for b in active for nid in books[b]['graph']['nodes']}

        class CaptureMerger(SemanticMerger):
            def _pair(self, a, b):
                pair = super()._pair(a, b)
                origins.setdefault(pair['pair_id'], set()).add((a['id'], b['id']))
                return pair

            def _judge(self, pairs):
                for pair in pairs:
                    count = report['counts']
                    count['attempted'] += 1
                    with runtime.lock:
                        if runtime.semantic_decision(pair['pair_id']) is not None:
                            count['preservedexisting'] += 1
                            continue
                    endpoints = origins[pair['pair_id']]
                    bids = {owner.get(nid) for endpoint in endpoints for nid in endpoint}
                    if len(bids) != 1 or None in bids or next(iter(bids)) not in by_book:
                        count['noneligible'] += 1
                        continue
                    source = by_book[next(iter(bids))]
                    native = [source.rebind(pair, *endpoint) for endpoint in sorted(endpoints)]
                    if any(p is None for p in native) or any(p != native[0] for p in native):
                        count['mismatch'] += 1
                        continue
                    original = native[0]
                    key, encoded = source.lookup(original)
                    if encoded is None:
                        count['nohit'] += 1
                        continue
                    try:
                        decision = json.loads(encoded)
                    except (json.JSONDecodeError, TypeError):
                        decision = None
                    if not _valid_decision(decision, original):
                        count['invalid'] += 1
                        report['invalid'].append(dict(pair_id=pair['pair_id'], source_pair_id=original['pair_id'], source_cache_key=key))
                        continue
                    rebound = dict(decision, pair_id=pair['pair_id'],
                                   evidence_ids=[source.maps['evidence'][eid] for eid in decision['evidence_ids']])
                    if not _valid_decision(rebound, pair):
                        raise ValueError('Rebound decision violated the original evidence contract')
                    # 同一运行时锁内检查后写入，不覆盖已经存在的判定。
                    with runtime.lock:
                        if runtime.semantic_decision(pair['pair_id']) is not None:
                            count['preservedexisting'] += 1
                            continue
                        runtime.save_semantic_decisions({pair['pair_id']:rebound})
                    count['hit'] += 1
                    report['hits'].append(dict(current_pair_id=pair['pair_id'], source_pair_id=original['pair_id'],
                        source_cache_key=key, source_cache=copy.deepcopy(source.record['cache']),
                        contribution=copy.deepcopy(source.record['contribution']), artifact_id=source.record['artifact_id'],
                        semantic_fingerprint=source.record['semantic_fingerprint'],
                        source_decision_hash=digest(decision), rebound_decision_hash=digest(rebound),
                        cost_origin='historical_source_cache_reuse'))

        origins = {}
        # 此处映射结果直接丢弃，之后由未改动的 build_view 正常执行融合。
        CaptureMerger(runtime, evidence, book_rows).deduplicate(dict(nodes=nodes, claims=claims))
    return report
