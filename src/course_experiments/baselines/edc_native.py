"""独立 EDC iter0 运行桥；保留上游代码、提示和关系规范化。"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import locale
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Callable
import urllib.error
import urllib.request


SOURCE_PIN = '02aad17e2fc1e9845c84746b384e7fd3160edc1a'
SOURCE_URL = 'https://github.com/clear-nus/edc'
BRIDGE_VERSION = 'edc-native-iter0-v1.2'
METHOD_VARIANT = 'EDC native + SD fullwidth relation separator compatibility'
BASE_URL = 'https://api.siliconflow.cn/v1'
MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
EMBEDDING_MODEL = 'BAAI/bge-m3'
TIMEOUT = 240
MAX_ATTEMPTS_PER_REQUEST = 3
SOURCE_BLOBS = {
    'edc/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'edc/edc_framework.py': 'abf708ad1966766f8f6e29aadb1a5ff3ed38697f',
    'edc/entity_extraction.py': '0d7cf690614f18456d506a9a90379fd44e287873',
    'edc/extract.py': '7b7755b5000576b3ecd1eb9a8627e689d356c1e0',
    'edc/schema_canonicalization.py': 'e76a28c19132384e3b37263adbc18543a76b1a82',
    'edc/schema_definition.py': '98a90b2963c9a97f9aef8c33a768fccc5d9a839e',
    'edc/schema_retriever.py': '8c69008421fb795f178d1b8dffbaf737f6517b8c',
    'edc/utils/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'edc/utils/e5_mistral_utils.py': 'f0376d7fcffd07b4c54c0799d42c6b7fd19e59e6',
    'edc/utils/llm_utils.py': 'ab17fe354fd1271784889d12c355a6a3e86b9450',
    'few_shot_examples/example/ee_few_shot_examples.txt': 'a64be3c3332ec708cefa3fce8ef9bec5a2c0d4f5',
    'few_shot_examples/example/oie_few_shot_examples.txt': '702ac093f4e8390ecd68679fafb04cc593d47a63',
    'few_shot_examples/example/oie_few_shot_refine_examples.txt': '503a59bc7b8d35448649bae83cc2f341e3ca2b73',
    'few_shot_examples/example/sd_few_shot_examples.txt': '711125d6e5e812e9a5bbc7cc08bf66c1fdc468a4',
    'prompt_templates/ee_template.txt': '7248170ada2fbceb3b3d81920e4dea82628bda75',
    'prompt_templates/em_template.txt': 'a1bd70cd10a8583d487638f369febe272efecc01',
    'prompt_templates/oie_r_template.txt': 'c217dd736e8cfbea30804c8c8b8cdf378058e017',
    'prompt_templates/oie_template.txt': '00316139689f7bf4bf42caaa19591b012ab74d80',
    'prompt_templates/sc_template.txt': 'babdf10733567b1a04e2f27250c1f448d4643b87',
    'prompt_templates/sd_template.txt': '1b15d22bc2a1aa0e4e571cfa71f1471082b5cd06',
}
DEPENDENCIES = ('numpy', 'torch', 'transformers', 'sentence_transformers', 'openai',
                'peft', 'tqdm', 'datasets', 'evaluate', 'accelerate', 'huggingface_hub',
                'sklearn', 'scipy', 'joblib', 'threadpoolctl')


class NativeRunError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(_json(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _bridge_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sd_separator_compat(text: str, relations: set[str]) -> tuple[str, list[dict]]:
    lines = text.splitlines(keepends=True)
    changes = []
    for index, line in enumerate(lines):
        prefix, separator, rest = line.partition('：')
        if separator and prefix.strip() in relations:
            lines[index] = prefix + ':' + rest
            changes.append({'line': index + 1, 'relation': prefix.strip(), 'from': '：', 'to': ':'})
    return ''.join(lines), changes


def synthetic_units() -> list[dict]:
    return [
        {'unit_id': 'synthetic_zh_001', 'text': '处理器包含运算器和控制器。处理器执行指令，运算器执行算术运算，控制器协调指令执行。处理器从存储器读取指令。', 'source_kind': 'synthetic'},
        {'unit_id': 'synthetic_zh_002', 'text': '中央处理器也称处理器。处理器执行指令，控制器包含指令译码器。存储器保存指令和数据。', 'source_kind': 'synthetic'},
    ]


def read_source_hashes(source_dir: Path) -> dict:
    result = {}
    for name, expected in SOURCE_BLOBS.items():
        data = (Path(source_dir) / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != expected:
            raise NativeRunError(f'pinned source bytes changed: {name}')
        result[name] = {'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest()}
    return result


def prepare(source_repo: Path, run_dir: Path, units: list[dict], *,
            max_logical_calls: int, max_http_attempts: int) -> dict:
    """只读取本地 Git 对象并冻结输入；不导入模型、不联网。"""
    if (type(max_logical_calls) is not int or max_logical_calls < 1 or
            type(max_http_attempts) is not int or max_http_attempts < 1 or
            max_http_attempts > max_logical_calls * MAX_ATTEMPTS_PER_REQUEST):
        raise ValueError('explicit positive logical/HTTP budgets required; HTTP <= logical * 3')
    if not units or any(not isinstance(u.get('unit_id'), str) or not u['unit_id'].strip()
                        or not isinstance(u.get('text'), str) or not u['text'].strip() for u in units):
        raise ValueError('nonempty unit_id and text required')
    if len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('duplicate unit_id')
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = run_dir / 'native_source'
    source_dir.mkdir(exist_ok=True)
    source_repo = Path(source_repo)
    source_bundle = (source_repo / 'edc/edc_framework.py').is_file()
    if source_bundle:
        read_source_hashes(source_repo)
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0')
    for name in SOURCE_BLOBS:
        destination = source_dir / name
        if not destination.exists():
            data = ((source_repo / name).read_bytes() if source_bundle else subprocess.check_output(
                ['git', '-C', str(source_repo), '-c', 'remote.origin.url=', '-c', 'remote.origin.promisor=false',
                 'show', f'{SOURCE_PIN}:{name}'], env=env))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    plan = {
        'bridge_version': BRIDGE_VERSION, 'bridge_sha256': _bridge_hash(),
        'method_variant': METHOD_VARIANT,
        'source_url': SOURCE_URL, 'source_pin': SOURCE_PIN,
        'source_files': read_source_hashes(source_dir), 'units': units,
        'input_sha256': _hash(units), 'model': MODEL, 'embedding_model': EMBEDDING_MODEL,
        'base_url': BASE_URL, 'temperature': 0, 'seed': 42,
        'native_max_tokens': {'oie': 512, 'sd': 512, 'sc': 1},
        'target_schema_path': None, 'enrich_schema': True, 'refinement_iterations': 0,
        'budget': {'max_logical_calls': max_logical_calls, 'max_http_attempts': max_http_attempts,
                   'max_attempts_per_request': MAX_ATTEMPTS_PER_REQUEST, 'timeout_seconds': TIMEOUT},
        'provenance': 'input index -> immutable unit manifest; paragraph only; no edge spans',
        'entity_types': 'unsupported', 'entity_canonicalization': 'not_added',
        'preparation_status': 'prepared_zero_model_calls', 'model_execution_status': 'not_executed',
    }
    plan['freeze_sha256'] = _hash(plan)
    path = run_dir / 'prepare.json'
    if path.exists() and _read(path) != plan:
        raise ValueError('run is frozen; use a new run directory for changed inputs/configuration')
    if not path.exists():
        _write(path, plan)
    return plan


@contextmanager
def _native_import(source_dir: Path):
    if any(n == 'edc' or n.startswith('edc.') for n in sys.modules):
        raise NativeRunError('EDC already imported; use an isolated process')
    previous_path = list(sys.path)
    offline_names = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    previous_env = {name: os.environ.get(name) for name in offline_names}
    sys.path.insert(0, str(source_dir.resolve()))
    for name in offline_names:
        os.environ[name] = '1'
    try:
        yield importlib.import_module('edc.edc_framework')
    finally:
        sys.path[:] = previous_path
        for name in list(sys.modules):
            if name == 'edc' or name.startswith('edc.'):
                del sys.modules[name]
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def dependency_preflight(source_dir: Path, *, actual_import: bool = False) -> dict:
    packages = {}
    for name in DEPENDENCIES:
        found = importlib.util.find_spec(name) is not None
        try:
            version = importlib.metadata.version(name.replace('_', '-')) if found else None
        except importlib.metadata.PackageNotFoundError:
            version = 'unknown'
        packages[name] = {'found': found, 'version': version}
    report = {'python': sys.version, 'packages': packages,
              'default_encoding': locale.getpreferredencoding(False),
              'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
              'native_import_status': 'not_attempted', 'model_execution_status': 'not_executed'}
    if actual_import:
        try:
            read_source_hashes(source_dir)
            with _native_import(source_dir) as framework:
                if not callable(framework.EDC.extract_kg):
                    raise NativeRunError('native entry missing')
            report['native_import_status'] = 'passed'
        except Exception as error:
            report['native_import_status'] = 'failed'
            report['error_type'] = type(error).__name__
            report['error'] = str(error)
    return report


class Transport:
    def __init__(self, run_dir: Path, plan: dict, api_key: str, *, sender: Callable | None = None):
        self.run_dir = Path(run_dir)
        self.plan = plan
        self.api_key = api_key
        self.sender = sender or self._send
        self.kind = 'offline_http_fixture' if sender is not None else 'live_siliconflow'
        self.path = self.run_dir / 'attempts.jsonl'
        self.events = []
        if self.path.exists():
            self.events = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
        if any(e['execution_kind'] != self.kind for e in self.events):
            raise NativeRunError('cannot mix offline fixtures with live request cache')

    def _append(self, event: dict):
        event['execution_kind'] = self.kind
        text = _json(event)
        if self.api_key:
            text = text.replace(self.api_key, '[REDACTED]')
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(text + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(json.loads(text))

    @staticmethod
    def _send(request, timeout):
        if request.full_url not in (BASE_URL + '/chat/completions', BASE_URL + '/embeddings'):
            raise NativeRunError('unapproved initial transport URL')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode('utf-8', errors='replace')

    def _cache_response(self, route: str, key: str, raw: str) -> dict:
        try:
            response = json.loads(raw)
            if route == 'chat/completions':
                content = response['choices'][0]['message']['content']
                if not isinstance(content, str) or not content:
                    raise ValueError('empty completion')
            elif not isinstance(response['data'], list) or not response['data']:
                raise ValueError('empty embedding response')
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self._append({'event': 'terminal_failure', 'key': key, 'error_type': type(error).__name__})
            raise NativeRunError('invalid model response; raw evidence retained') from None
        self._append({'event': 'success', 'key': key, 'response_raw': raw})
        return response

    def sd_response_compatibility(self, original: str, relations: set[str], parser: Callable, body: dict) -> str:
        transformed, changes = _sd_separator_compat(original, relations)
        audit = {'event': 'sd_separator_compatibility', 'stage': 'sd', 'method_variant': METHOD_VARIANT,
                 'original_text': original, 'transformed_text': transformed, 'changes': changes,
                 'expected_relations': sorted(relations), 'request_body_sha256': _hash(body)}
        audit['key'] = 'sd-compat:' + _hash(audit)
        if not any(e['event'] == audit['event'] and e['key'] == audit['key'] for e in self.events):
            self._append(audit)
        missing = relations - set(parser(transformed))
        if missing:
            raise NativeRunError('missing SD definitions after separator compatibility: ' + repr(sorted(missing)))
        return transformed

    def request(self, route: str, body: dict, stage: str) -> dict:
        expected_model = EMBEDDING_MODEL if route == 'embeddings' else MODEL
        if route not in ('embeddings', 'chat/completions') or body.get('model') != expected_model:
            raise NativeRunError('unapproved model or route')
        identity = {'pin': self.plan['source_pin'], 'source': self.plan['source_files'],
                    'bridge': self.plan['bridge_sha256'], 'version': BRIDGE_VERSION,
                    'input_sha256': self.plan['input_sha256'], 'base_url': BASE_URL,
                    'stage': stage, 'route': route, 'body': body, 'execution_kind': self.kind}
        key = _hash(identity)
        if any(e.get('status') == 402 for e in self.events):
            raise NativeRunError('402 balance stop persisted; no automatic restart')
        previous = [e for e in self.events if e['key'] == key]
        for event in previous:
            if event['event'] == 'success':
                return json.loads(event['response_raw'])
        starts = [e for e in previous if e['event'] == 'attempt_started']
        ends = [e for e in previous if e['event'] == 'attempt_finished']
        if len(starts) != len(ends):
            raise NativeRunError('unresolved in-flight request; refusing possible double charge')
        if any(e['event'] == 'terminal_failure' for e in previous):
            raise NativeRunError('terminal request failure persisted')
        for event in ends:
            if event['status'] == 200:
                return self._cache_response(route, key, event['response_raw'])
        all_starts = [e for e in self.events if e['event'] == 'attempt_started']
        logical_keys = {e['key'] for e in all_starts}
        budget = self.plan['budget']
        if key not in logical_keys and len(logical_keys) >= budget['max_logical_calls']:
            raise NativeRunError('logical call budget exhausted')
        for attempt in range(len(starts) + 1, MAX_ATTEMPTS_PER_REQUEST + 1):
            if len([e for e in self.events if e['event'] == 'attempt_started']) >= budget['max_http_attempts']:
                raise NativeRunError('HTTP attempt budget exhausted')
            self._append({'event': 'attempt_started', 'key': key, 'attempt': attempt, 'stage': stage,
                          'url': BASE_URL + '/' + route, 'request_body': body,
                          'timeout_seconds': TIMEOUT})
            request = urllib.request.Request(BASE_URL + '/' + route, data=_json(body).encode('utf-8'),
                                             headers={'Content-Type': 'application/json',
                                                      'Authorization': 'Bearer ' + self.api_key})
            try:
                status, raw = self.sender(request, TIMEOUT)
            except (OSError, TimeoutError) as error:
                self._append({'event': 'attempt_finished', 'key': key, 'attempt': attempt,
                              'status': None, 'error_type': type(error).__name__})
                continue
            self._append({'event': 'attempt_finished', 'key': key, 'attempt': attempt,
                          'status': status, 'response_raw': raw})
            if status == 200:
                return self._cache_response(route, key, raw)
            if status == 402 or status not in (408, 409, 429, 500, 502, 503, 504):
                self._append({'event': 'terminal_failure', 'key': key, 'status': status})
                raise NativeRunError(f'HTTP {status}; terminal failure retained')
        raise NativeRunError('request failed after at most 3 HTTP attempts')


class BGEEmbedder:
    prompts: dict = {}

    def __init__(self, transport: Transport):
        self.transport = transport
        self.dimension = None

    def encode(self, sentences, prompt_name=None, prompt=None, normalize_embeddings=True):
        import numpy as np
        if prompt_name is not None or prompt is not None:
            raise ValueError('BGE native iter0 bridge uses no additional query prompt')
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts or any(not isinstance(t, str) or not t for t in texts):
            raise ValueError('nonempty string embedding inputs required')
        response = self.transport.request('embeddings', {'model': EMBEDDING_MODEL, 'input': texts,
                                                       'encoding_format': 'float'}, 'sc_embedding')
        data = sorted(response['data'], key=lambda row: row['index'])
        if [row['index'] for row in data] != list(range(len(texts))):
            raise NativeRunError('embedding indices/count mismatch')
        try:
            vectors = np.asarray([row['embedding'] for row in data], dtype=np.float64)
        except (ValueError, KeyError, TypeError):
            raise NativeRunError('invalid embedding array') from None
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1 or not np.isfinite(vectors).all():
            raise NativeRunError('invalid embedding shape/values')
        if self.dimension is not None and vectors.shape[1] != self.dimension:
            raise NativeRunError('embedding dimension changed')
        self.dimension = vectors.shape[1]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise NativeRunError('zero embedding vector')
        if normalize_embeddings:
            vectors = vectors / norms
        return vectors[0] if single else vectors


@contextmanager
def native_bridge(framework, transport: Transport):
    """仅替换三个模型边界；上下文退出后恢复。"""
    llm_utils = importlib.import_module('edc.utils.llm_utils')
    original = (llm_utils.is_model_openai, llm_utils.openai_chat_completion, framework.SentenceTransformer)
    def route(model):
        if model != MODEL:
            raise NativeRunError('unexpected native model; local model download prohibited')
        return True
    def completion(model, system_prompt, history, temperature=0, max_tokens=512):
        route(model)
        callers = {frame.function for frame in inspect.stack()}
        stage = next((name for function, name in (('llm_verify', 'sc'), ('define_schema', 'sd'), ('extract', 'oie'))
                      if function in callers), None)
        if stage is None or max_tokens != transport.plan['native_max_tokens'][stage] or temperature != 0:
            raise NativeRunError('unexpected native stage/token/temperature contract')
        messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt is not None else []) + history
        body = {'model': model, 'messages': messages, 'temperature': temperature,
                'max_tokens': max_tokens, 'seed': 42}
        response = transport.request('chat/completions', body, stage)
        content = response['choices'][0]['message']['content']
        if stage == 'sd':
            native_frame = next((frame.frame for frame in inspect.stack()
                                 if frame.function == 'define_schema' and
                                 frame.frame.f_globals.get('__name__') == 'edc.schema_definition'), None)
            relations = native_frame.f_locals.get('relations_present') if native_frame is not None else None
            if not isinstance(relations, set) or not all(isinstance(r, str) for r in relations):
                raise NativeRunError('SD compatibility requires native relation context')
            content = transport.sd_response_compatibility(content, relations, llm_utils.parse_relation_definition, body)
        return content
    def embedder(model, **kwargs):
        if model != EMBEDDING_MODEL:
            raise NativeRunError('unexpected native embedder; download prohibited')
        return BGEEmbedder(transport)
    llm_utils.is_model_openai = route
    llm_utils.openai_chat_completion = completion
    framework.SentenceTransformer = embedder
    try:
        yield
    finally:
        llm_utils.is_model_openai, llm_utils.openai_chat_completion, framework.SentenceTransformer = original


def _configuration(source_dir: Path) -> dict:
    config = {f'{stage}_llm': MODEL for stage in ('oie', 'sd', 'sc', 'ee')}
    config.update({'sc_embedder': EMBEDDING_MODEL, 'sr_embedder': EMBEDDING_MODEL, 'sr_adapter_path': None,
                   'target_schema_path': None, 'enrich_schema': True, 'loglevel': logging.WARNING})
    for prefix, filename in (('oie', 'oie_template'), ('sd', 'sd_template'), ('sc', 'sc_template'),
                             ('ee', 'ee_template'), ('oie_refine', 'oie_r_template'), ('em', 'em_template')):
        config[prefix + '_prompt_template_file_path'] = str(source_dir / 'prompt_templates' / (filename + '.txt'))
    for prefix, filename in (('oie', 'oie'), ('sd', 'sd'), ('ee', 'ee'), ('oie_refine', 'oie_few_shot_refine')):
        name = filename + '_few_shot_examples.txt' if prefix != 'oie_refine' else 'oie_few_shot_refine_examples.txt'
        config[prefix + '_few_shot_example_file_path'] = str(source_dir / 'few_shot_examples/example' / name)
    return config


def _validate_artifacts(output_dir: Path, units: list[dict]) -> dict:
    stage_file = output_dir / 'iter0/result_at_each_stage.json'
    kg_file = output_dir / 'iter0/canon_kg.txt'
    rows = _read(stage_file)
    lines = kg_file.read_text(encoding='utf-8').splitlines()
    if len(rows) != len(units) or len(lines) != len(units):
        raise NativeRunError('native output unit coverage mismatch')
    predicates = set()
    for index, (row, unit, line) in enumerate(zip(rows, units, lines)):
        if row['index'] != index or row['input_text'] != unit['text']:
            raise NativeRunError('native input index/text mismatch')
        if any(key not in row for key in ('oie', 'schema_definition', 'canonicalization_candidates', 'schema_canonicalizaiton')):
            raise NativeRunError('native stages missing')
        triples = row['schema_canonicalizaiton']
        if not row['oie'] or not row['schema_definition'] or not triples or any(t is None for t in triples):
            raise NativeRunError('empty or null native stage output')
        if ast.literal_eval(line) != triples:
            raise NativeRunError('native JSON/text output disagree')
        if len(triples) != len(row['oie']):
            raise NativeRunError('native triple coverage mismatch')
        for original, triple in zip(row['oie'], triples):
            if len(triple) != 3 or not all(isinstance(v, str) and v for v in triple):
                raise NativeRunError('invalid native triple')
            if original[0] != triple[0] or original[2] != triple[2]:
                raise NativeRunError('native entity endpoints changed')
            predicates.add(triple[1])
    return {'unit_count': len(rows), 'distinct_predicates': len(predicates),
            'native_output_sha256': {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (stage_file, kg_file)},
            'unit_lineage': [{'index': i, 'unit_id': u['unit_id'], 'text_sha256': hashlib.sha256(u['text'].encode()).hexdigest()}
                             for i, u in enumerate(units)]}


def execute(run_dir: Path, *, api_key: str | None = None, sender: Callable | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    plan = _read(run_dir / 'prepare.json')
    if plan.get('freeze_sha256') != _hash({k: v for k, v in plan.items() if k != 'freeze_sha256'}):
        raise NativeRunError('plan freeze mismatch')
    if plan['bridge_sha256'] != _bridge_hash() or plan['source_files'] != read_source_hashes(run_dir / 'native_source'):
        raise NativeRunError('bridge/source freeze mismatch')
    if plan['input_sha256'] != _hash(plan['units']):
        raise NativeRunError('input freeze mismatch')
    if locale.getpreferredencoding(False).lower().replace('-', '') != 'utf8':
        raise NativeRunError('native prompts require UTF-8 default encoding; launch python -X utf8')
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise NativeRunError('launch Python with PYTHONHASHSEED=42 for native set/prompt reproducibility')
    key = api_key if api_key is not None else os.environ.get('SILICONFLOW_API_KEY', '')
    if not key:
        raise NativeRunError('SILICONFLOW_API_KEY is required')
    lock_path = run_dir / 'execute.lock'
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise NativeRunError('execution lock exists; verify no active process before manual recovery') from None
    os.close(descriptor)
    proof = {'bridge_version': BRIDGE_VERSION, 'bridge_sha256': plan['bridge_sha256'],
             'method_variant': METHOD_VARIANT,
             'source_pin': SOURCE_PIN, 'input_sha256': plan['input_sha256'],
             'execution_kind': 'offline_http_fixture' if sender is not None else 'live_siliconflow',
             'native_artifact_status': 'failed', 'model_execution_status': 'not_executed',
             'provenance': plan['provenance'], 'entity_types': 'unsupported'}
    previous_random = random.getstate()
    try:
        preflight = dependency_preflight(run_dir / 'native_source', actual_import=True)
        _write(run_dir / 'dependency_preflight.json', preflight)
        if preflight['native_import_status'] != 'passed':
            raise NativeRunError('native dependency import failed; see dependency_preflight.json')
        transport = Transport(run_dir, plan, key, sender=sender)
        output_index = 1
        while (run_dir / f'native_attempt_{output_index:04d}').exists():
            output_index += 1
        output_dir = run_dir / f'native_attempt_{output_index:04d}'
        proof['native_output_dir'] = str(output_dir)
        random.seed(42)
        with _native_import(run_dir / 'native_source') as framework, native_bridge(framework, transport):
            native = framework.EDC(**_configuration(run_dir / 'native_source'))
            native.extract_kg([u['text'] for u in plan['units']], str(output_dir), refinement_iterations=0)
        proof.update(_validate_artifacts(output_dir, plan['units']))
        proof['request_journal_sha256'] = hashlib.sha256(transport.path.read_bytes()).hexdigest()
        starts = [e for e in transport.events if e['event'] == 'attempt_started']
        proof['logical_requests'] = len({e['key'] for e in starts})
        proof['http_attempts'] = len(starts)
        proof['native_artifact_status'] = 'passed'
        proof['model_execution_status'] = 'succeeded' if sender is None else 'not_executed'
        proof['dependency_status'] = 'actual_native_import_passed'
        _write(run_dir / 'executionproof.json', proof)
        return proof
    except Exception as error:
        proof['error_type'] = type(error).__name__
        proof['error'] = str(error).replace(key, '[REDACTED]')
        if sender is None and (run_dir / 'attempts.jsonl').exists():
            proof['model_execution_status'] = 'failed'
        _write(run_dir / 'executionproof.json', proof)
        raise NativeRunError(f"{proof['error_type']}: {proof['error']}") from None
    finally:
        random.setstate(previous_random)
        lock_path.unlink()


def gate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    plan = _read(run_dir / 'prepare.json')
    proof_path = run_dir / 'executionproof.json'
    if not proof_path.exists():
        return {'native_artifact_status': 'not_executed', 'live_model_passed': False}
    proof = _read(proof_path)
    valid = proof['native_artifact_status'] == 'passed'
    if valid:
        if (plan.get('freeze_sha256') != _hash({k: v for k, v in plan.items() if k != 'freeze_sha256'}) or
                proof['bridge_sha256'] != _bridge_hash() or proof['input_sha256'] != _hash(plan['units'])):
            raise NativeRunError('execution proof freeze mismatch')
        if proof['request_journal_sha256'] != hashlib.sha256((run_dir / 'attempts.jsonl').read_bytes()).hexdigest():
            raise NativeRunError('execution request journal changed')
        read_source_hashes(run_dir / 'native_source')
        checked = _validate_artifacts(Path(proof['native_output_dir']), plan['units'])
        if any(proof[k] != checked[k] for k in checked):
            raise NativeRunError('execution artifact evidence changed')
    return {**proof, 'live_model_passed': valid and proof['execution_kind'] == 'live_siliconflow'
            and proof['model_execution_status'] == 'succeeded'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent pinned native EDC iter0 smoke bridge')
    parser.add_argument('action', choices=('prepare', 'preflight', 'execute', 'gate'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--units', type=Path)
    parser.add_argument('--max-logical-calls', type=int)
    parser.add_argument('--max-http-attempts', type=int)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if args.source_repo is None or args.max_logical_calls is None or args.max_http_attempts is None:
            parser.error('prepare requires source-repo and both explicit budgets')
        result = prepare(args.source_repo, args.run_dir, _read(args.units) if args.units else synthetic_units(),
                         max_logical_calls=args.max_logical_calls, max_http_attempts=args.max_http_attempts)
    elif args.action == 'preflight':
        result = dependency_preflight(args.run_dir / 'native_source', actual_import=True)
        _write(args.run_dir / 'dependency_preflight.json', result)
    elif args.action == 'execute':
        result = execute(args.run_dir)
    else:
        result = gate(args.run_dir)
    print(_json(result))
    if args.action == 'gate':
        return 0 if result['live_model_passed'] else 2
    if args.action == 'preflight':
        return 0 if result['native_import_status'] == 'passed' else 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
