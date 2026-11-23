from __future__ import annotations
from pathlib import Path
from domain_ckg_workspace.ckg_incremental.engine import IncrementalEngine
from .contributions import export_contribution, write_immutable, read_contribution
from .manifest import digest


def generate_prior(course_id: str, description: str, schema: dict, config: dict,
                   runtime, output: Path, input_id: str) -> dict:
    output = Path(output)
    artifact = output / 'contribution.json'
    if artifact.exists():
        old = read_contribution(artifact)
        if old['input_id'] != input_id: raise ValueError('Frozen G0 input changed')
        return old
    config = dict(config, domain=description, prior_concepts=60, prior_relations=100)
    corpus = dict(books=[], evidence=[], chunks=[])
    corpus['fingerprint'] = digest(corpus)
    engine = IncrementalEngine(corpus,output,schema,config,runtime)
    try:
        engine.seed()
        result = export_contribution(engine.store.snapshot(),course_id,None,'full',input_id)
        return write_immutable(artifact,result)
    finally:
        engine.store.close()
