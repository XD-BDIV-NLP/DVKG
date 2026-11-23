from __future__ import annotations
from pathlib import Path
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable

ROOT=Path(__file__).resolve().parents[2]
def assets():
    paths=list((ROOT/'course_experiments/evaluation').glob('*.py'))+[ROOT/'course_experiments/splits.py',ROOT/'course_experiments/comparison.py',ROOT/'course_experiments/manifest.py',ROOT/'course_experiments/contributions.py']
    paths+=list((ROOT/'prompts/evaluation').glob('*.txt'))
    return {str(p.relative_to(ROOT)):p for p in paths}

def freeze_phase(paths,parameters,output):
    if parameters.get('seed')!=42:raise ValueError('Fixed seed42 required')
    files={k:file_record(v) for k,v in dict(paths,**assets()).items()}
    payload=dict(format_version=1,parameters=parameters,files=files)
    payload['phase_id']=digest(payload);write_immutable(output,payload);return payload

def validate_phase(path):
    p=read_document(path)
    if p['phase_id']!=digest({k:v for k,v in p.items() if k!='phase_id'}):raise ValueError('Phase digest mismatch')
    for f in p['files'].values():
        if file_record(f['path'])!=f:raise ValueError('Phase asset changed: '+f['path'])
    covered={(f['path'],f['sha256']) for f in p['files'].values()}
    if any((r['path'],r['sha256']) not in covered for r in map(file_record,assets().values())):raise ValueError('Actual evaluation implementation not frozen')
    return p

def bound_input(phase,role,value):
    if role not in phase['files'] or read_document(phase['files'][role]['path'])!=value:raise ValueError('Input not bound to phase: '+role)

def check_llm(options):
    if options.get('base_url','').rstrip('/')!='https://api.siliconflow.cn/v1' or options.get('model')!='deepseek-ai/DeepSeek-V4-Flash' or options.get('extra_body',{}).get('enable_thinking') is not False:raise ValueError('Official frozen LLM with enable_thinking:false required')

def check_runtime(config):
    check_llm(config['llm'])
    if config.get('embedding',{}).get('base_url','').rstrip('/')!='https://api.siliconflow.cn/v1' or config['embedding'].get('model')!='BAAI/bge-m3':raise ValueError('Official frozen BGE-M3 required')

def bind_runtime(phase,runtime):
    check_runtime(runtime.config)
    if 'runtime_config' in phase['files']:expected=read_document(phase['files']['runtime_config']['path'])
    elif 'construction' in phase['files']:
        construction=read_document(phase['files']['construction']['path']);record=construction['files']['assets']['runtime_config']
        if file_record(record['path'])!=record:raise ValueError('Construction runtime configuration changed')
        expected=read_document(record['path'])
    else:raise ValueError('Actual runtime configuration must be frozen')
    if runtime.config!=expected:raise ValueError('Actual runtime differs from frozen configuration')
    return digest(runtime.config)

def runtime_for(construction,root):
    from ..construction import PROMPTS
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    config=read_document(construction['files']['assets']['runtime_config']['path'])
    check_runtime(config)
    return ModelRuntime(root,PROMPTS,config)
