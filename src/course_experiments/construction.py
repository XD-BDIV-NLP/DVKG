from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime, save
from .manifest import digest, file_record, read_document, require_construction
from .contributions import export_contribution, write_immutable, read_contribution
from .prior import generate_prior
from .textbook import TextbookEngine, scoped_corpus

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / 'prompts/incremental_v5'


def implementation_assets() -> dict[str, Path]:
    files = list((ROOT / 'domain_ckg_workspace/ckg_incremental').glob('*.py'))
    files += [ROOT / 'course_experiments' / name for name in
              ('manifest.py','prior.py','textbook.py','contributions.py','construction.py')]
    files += [PROMPTS / name for name in {n for pair in ModelRuntime.FILES.values() for n in pair}]
    return {'implementation_' + str(p.relative_to(ROOT)).replace('/','_').replace('\\','_'): p for p in files}


def check_config(config: dict) -> None:
    expected = dict(workers=8, prior_concepts=60, prior_relations=100, service_retries=6)
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Construction requires workers8, prior60/100, retries6')
    if any(config.get('llm',{}).get(k) != v for k,v in
           dict(model='deepseek-ai/DeepSeek-V4-Flash',temperature=0,max_tokens=12000,timeout_seconds=240).items()):
        raise ValueError('Construction LLM settings differ from fixed protocol')
    if config.get('embedding',{}).get('model') != 'BAAI/bge-m3' or config['embedding'].get('timeout_seconds') != 180:
        raise ValueError('Construction embedding settings differ from fixed protocol')


def validate_inputs(manifest: Path, course_id: str) -> tuple[dict, dict, dict, dict]:
    frozen = require_construction(manifest)
    assets = frozen['files']['assets']
    if 'runtime_config' not in assets:
        raise ValueError('Frozen runtime_config asset required')
    covered = {(r['path'],r['sha256']) for r in assets.values()}
    for path in implementation_assets().values():
        record = file_record(path)
        if (record['path'],record['sha256']) not in covered:
            raise ValueError(f'Actual implementation/prompt not frozen: {path}')
    config = read_document(assets['runtime_config']['path'])
    check_config(config)
    budgets = frozen['experiments']['budgets']
    if any(budgets[role]['max_output_tokens'] != 12000 or budgets[role]['temperature'] != 0 for role in ('g0','textbook')):
        raise ValueError('Frozen model budget mismatch')
    textbook = budgets['textbook']
    if textbook['max_retries'] != 6 or any(textbook[k] != config[k] for k in ('chunk_chars','evidence_chars')):
        raise ValueError('Frozen textbook budget mismatch')
    courses = [c for c in frozen['courses']['courses'] if c['course_id']==course_id]
    if len(courses)!=1: raise ValueError('Unknown course_id')
    return frozen,courses[0],config,read_document(assets['schema']['path'])


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True,exist_ok=True)
    with (root / '.construction.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1): stream.write(b' '); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            try: yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield


def construct(manifest: Path, course_id: str, action: str, output_root: Path,
              book_alias: str | None = None, smoke_chunks: int | None = None) -> dict:
    frozen,course,config,schema = validate_inputs(manifest,course_id)
    config = dict(config,domain=course['description'])
    if action == 'prior':
        if smoke_chunks is not None or book_alias is not None:
            raise ValueError('Fixed G0 has no book or smoke input')
        identity = digest(dict(g0=frozen['g0_identities'][course_id],
            assets=frozen['files']['assets'],config=config))
        root = Path(output_root)/course_id/'g0'
        corpus = None
    elif action == 'textbook':
        books = [b for b in course['books'] if b['alias']==book_alias]
        if len(books)!=1: raise ValueError('Textbook requires a declared book alias')
        book_id = books[0]['book_id']
        corpus = scoped_corpus(read_document(frozen['files']['sources'][book_id]['path']),smoke_chunks)
        identity = digest(dict(manifest=frozen['manifest_id'],book_id=book_id,corpus=corpus['fingerprint']))
        root = Path(output_root)/course_id/book_id/corpus['construction_scope']/identity
    else: raise ValueError('Unknown construction action')
    runtime = engine = None
    with output_lock(root):
        metadata = dict(input_id=identity,scope='full' if corpus is None else corpus['construction_scope'],
                        course_id=course_id,book_id=None if corpus is None else book_id)
        write_immutable(root/'construction_identity.json',metadata)
        artifact = root/'contribution.json'
        if artifact.exists():
            old = read_contribution(artifact)
            if old['input_id'] != identity: raise ValueError('Artifact identity mismatch')
            return old
        try:
            # 加锁和读取输入后再次校验，再创建仅属于当前构建的请求缓存。
            validate_inputs(manifest,course_id)
            runtime = ModelRuntime(root/'runtime',PROMPTS,config)
            if corpus is None:
                return generate_prior(course_id,course['description'],schema,config,runtime,root,identity)
            engine = TextbookEngine(corpus,root,schema,config,runtime)
            engine.run()
            incomplete = [c['id'] for c in corpus['chunks'] if
                          (engine.store.chunk_state(c['id']) or {}).get('status') != 'complete']
            if incomplete:
                raise RuntimeError(f'Unfinished textbook chunks: {incomplete}')
            result = export_contribution(engine.store.snapshot(),course_id,book_id,
                                         corpus['construction_scope'],identity)
            return write_immutable(artifact,result)
        except Exception as exc:
            save(root/'construction_error.json',dict(error_type=type(exc).__name__,message=str(exc)))
            raise
        finally:
            if engine is not None: engine.store.close()
            if runtime is not None: runtime.close()


def main():
    parser = argparse.ArgumentParser(description='Frozen independent course graph construction')
    parser.add_argument('action',choices=('prior','textbook'))
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--course-id',required=True)
    parser.add_argument('--book-alias',choices=('A','B','C','D'))
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--smoke-chunks',type=int)
    args = parser.parse_args()
    result = construct(args.manifest,args.course_id,args.action,args.output_root,args.book_alias,args.smoke_chunks)
    print(result['artifact_id'])


if __name__ == '__main__':
    main()
