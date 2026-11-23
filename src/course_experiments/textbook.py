from __future__ import annotations
import copy
from domain_ckg_workspace.ckg_incremental.engine import IncrementalEngine
from .manifest import digest

class TextbookEngine(IncrementalEngine):
    def seed(self):
        return None

    def complete_prior(self):
        return 0


def scoped_corpus(corpus: dict, smoke_chunks: int | None) -> dict:
    if smoke_chunks is not None and (type(smoke_chunks) is not int or smoke_chunks < 1):
        raise ValueError('smoke_chunks must be positive')
    result = copy.deepcopy(corpus)
    if smoke_chunks is not None:
        result['chunks'] = result['chunks'][:smoke_chunks]
        ids = {eid for c in result['chunks'] for key in ('evidence_ids','context_evidence_ids') for eid in c.get(key,[])}
        result['evidence'] = [e for e in result['evidence'] if e['id'] in ids]
    result['construction_scope'] = 'full' if smoke_chunks is None else 'smoke'
    result['source_corpus_fingerprint'] = corpus['fingerprint']
    result['smoke_chunks'] = smoke_chunks
    result['fingerprint'] = digest({k:v for k,v in result.items() if k!='fingerprint'})
    return result
