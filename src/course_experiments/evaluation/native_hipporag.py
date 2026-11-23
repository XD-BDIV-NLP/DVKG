"""官方 HippoRAG 2 薄适配；教材、原始收据与索引只应在原服务器使用。"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from filelock import FileLock
import numpy as np


SOURCE_URL = 'https://github.com/OSU-NLP-Group/HippoRAG'
SOURCE_PIN = 'd5c8329422e0a0b834a15874545cb6a74b4f9b26'
BRIDGE_VERSION = 'hipporag2-native-siliconflow-v1.1-openie-json'
BASE_URL = 'https://api.siliconflow.cn/v1'
LLM_MODEL = 'deepseek-ai/DeepSeek-V4-Flash'
EMBEDDING_MODEL = 'BAAI/bge-m3'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def verify_source(source_root):
    root = Path(source_root).resolve()
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RuntimeError('Pinned HippoRAG source checkout is required') from exc
    if git('rev-parse', 'HEAD') != SOURCE_PIN:
        raise RuntimeError('HippoRAG commit differs from frozen upstream')
    if git('status', '--porcelain', '--untracked-files=all', '--', 'src/hipporag', 'pyproject.toml'):
        raise RuntimeError('HippoRAG source or dependency declaration has changed')
    return root


def _check_config(config):
    if config.llm_name != LLM_MODEL or config.llm_base_url != BASE_URL:
        raise ValueError('Official frozen Flash configuration is required')
    if config.embedding_model_name != EMBEDDING_MODEL or config.embedding_base_url != BASE_URL:
        raise ValueError('Official frozen BGE-M3 configuration is required')


def _usage(raw, chat=False):
    usage = raw.get('usage')
    keys = ['prompt_tokens', 'total_tokens'] + (['completion_tokens'] if chat else [])
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in keys):
        raise ValueError('Standard usage is required for native accounting')
    return usage


class _AuditedComponent:
    def __init__(self, global_config, request_executor, audit_path, role):
        _check_config(global_config)
        self.global_config = global_config
        self.request_executor = request_executor
        self.audit_path = Path(audit_path)
        self.role = role
        self._error = None
        self._lock = threading.Lock()

    def check_error(self):
        if self._error is not None:
            raise self._error

    def _audit(self, record):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.audit_path) + '.lock'):
            with self.audit_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def _request(self, endpoint, body, validate):
        self.check_error()
        _check_config(self.global_config)
        stage = 'native_hipporag/' + self.role
        request_id = _digest(dict(endpoint=endpoint, base_url=BASE_URL, body=body))
        started = time.monotonic()
        try:
            raw, cache_hit = self.request_executor(endpoint, body, stage)
            value = validate(raw)
        except Exception as exc:
            # 上游识别记忆会捕获异常；保留错误，阶段末必须重新抛出。
            with self._lock:
                if self._error is None:
                    self._error = exc
            self._audit(dict(request_id=request_id, stage=stage, status='failed',
                error_type=type(exc).__name__, max_tokens=body.get('max_tokens'),
                elapsed_seconds=time.monotonic() - started))
            raise
        self._audit(dict(request_id=request_id, stage=stage, status='completed',
            cache_hit=bool(cache_hit), max_tokens=body.get('max_tokens'), usage=raw['usage'],
            finish_reason=raw.get('choices', [{}])[0].get('finish_reason'),
            elapsed_seconds=time.monotonic() - started))
        return value, bool(cache_hit)

    def close(self):
        pass


class AuditedLLM(_AuditedComponent):
    def __init__(self, global_config, request_executor, audit_path, role='extraction_llm'):
        super().__init__(global_config, request_executor, audit_path, role)
        self.llm_name = LLM_MODEL

    def infer(self, chat=None, **kwargs):
        messages = kwargs.pop('messages', chat)
        if not isinstance(messages, list) or not messages:
            raise ValueError('Native messages must be a nonempty list')
        aliases = [key for key in ('max_new_tokens', 'max_tokens', 'max_completion_tokens', 'max_output_tokens') if key in kwargs]
        if len(aliases) > 1:
            raise ValueError('Only one generation token parameter is allowed')
        limit = kwargs.pop(aliases[0]) if aliases else self.global_config.max_new_tokens
        if type(limit) is not int or limit < 1:
            raise ValueError('Native generation limit must be positive')
        body = dict(model=LLM_MODEL, messages=messages, temperature=0, seed=42,
            enable_thinking=False, max_tokens=limit)
        if 'response_format' in kwargs:
            value = kwargs.pop('response_format')
            if value is not None:
                body['response_format'] = value
        if kwargs:
            raise ValueError('Unsupported native generation parameters: ' + ','.join(sorted(kwargs)))
        def validate(raw):
            usage = _usage(raw, chat=True)
            choice = raw['choices'][0]
            if choice.get('finish_reason') not in {'stop', 'length'}:
                raise ValueError('Native completion did not finish normally')
            text = choice['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Native completion content is empty')
            return text, dict(usage, finish_reason=choice['finish_reason'])
        (text, metadata), cache_hit = self._request('/chat/completions', body, validate)
        return text, metadata, cache_hit


class AuditedEmbedding(_AuditedComponent):
    query_instruction_mode = 'ignored'
    embedding_dim = 1024
    embedding_model_name = EMBEDDING_MODEL

    def __init__(self, global_config, request_executor, audit_path):
        super().__init__(global_config, request_executor, audit_path, 'embedding')

    def batch_encode(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError('Embedding inputs must be strings')
        normalize = kwargs.pop('norm', self.global_config.embedding_return_as_normalized)
        batch_size = kwargs.pop('batch_size', getattr(self.global_config, 'embedding_batch_size', 16))
        instruction = kwargs.pop('instruction', None)
        if instruction:
            raise ValueError('BGE-M3 API adapter does not prepend query instructions')
        if kwargs or type(normalize) is not bool or type(batch_size) is not int or batch_size < 1:
            raise ValueError('Unsupported embedding parameters')
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if len(texts) > batch_size:
            return np.concatenate([self.batch_encode(texts[offset:offset + batch_size],
                norm=normalize, batch_size=batch_size) for offset in range(0, len(texts), batch_size)])
        body = dict(model=EMBEDDING_MODEL, input=texts, encoding_format='float')
        def validate(raw):
            _usage(raw)
            rows = raw['data']
            if len(rows) != len(texts) or {r.get('index') for r in rows} != set(range(len(texts))):
                raise ValueError('Embedding response indices do not match inputs')
            array = np.asarray([r['embedding'] for r in sorted(rows, key=lambda r: r['index'])], dtype=np.float32)
            if array.shape != (len(texts), self.embedding_dim) or not np.isfinite(array).all():
                raise ValueError('Frozen BGE-M3 requires finite 1024-dimensional vectors')
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError('Zero embedding vector')
            return array / norms if normalize else array
        vectors, _ = self._request('/embeddings', body, validate)
        return vectors

    def get_query_doc_scores(self, query_vec, doc_vecs):
        return np.dot(query_vec, doc_vecs.T)


def prepare_documents(rows, target_book_id, allowed_book_ids):
    if target_book_id not in allowed_book_ids:
        raise ValueError('Target book is not in the frozen allowlist')
    groups = {}
    count = 0
    for row in rows:
        if row.get('book_id') != target_book_id:
            raise ValueError('Foreign textbook row in native target index')
        text = row.get('text')
        ids = row.get('evidence_ids')
        if not isinstance(text, str) or not text.strip() or not isinstance(ids, list) or not ids or any(not isinstance(e, str) or not e for e in ids):
            raise ValueError('Target text and evidence identifiers are required')
        key = _digest(text)
        record = groups.setdefault(key, dict(text=text, source_id=target_book_id + ':' + key,
            metadata=dict(book_id=target_book_id, evidence_ids=[], chapter_ids=[])))
        record['metadata']['evidence_ids'].extend(ids)
        if row.get('chapter_id'):
            record['metadata']['chapter_ids'].append(row['chapter_id'])
        count += 1
    if not count:
        raise ValueError('Target corpus is empty')
    docs = [groups[key] for key in sorted(groups)]
    for doc in docs:
        for key in ('evidence_ids', 'chapter_ids'):
            doc['metadata'][key] = sorted(set(doc['metadata'][key]))
    identity = dict(target_book_id=target_book_id, row_count=count,
        unique_passage_count=len(docs), corpus_digest=_digest(docs),
        passage_digests=sorted(groups))
    return docs, identity


class NativeHippoRAG:
    def __init__(self, *, source_root, save_dir, target_book_id, allowed_book_ids, request_executor):
        if target_book_id not in allowed_book_ids:
            raise ValueError('Target book is not in the frozen allowlist')
        self.source_root = verify_source(source_root)
        src = self.source_root / 'src'
        existing = sys.modules.get('hipporag')
        expected = src / 'hipporag' / '__init__.py'
        if existing is not None and Path(existing.__file__).resolve() != expected:
            raise RuntimeError('Another HippoRAG source is already imported')
        sys.path.insert(0, str(src))
        module = importlib.import_module('hipporag')
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('HippoRAG import did not resolve to the pinned source')
        BaseConfig = importlib.import_module('hipporag.utils.config_utils').BaseConfig
        self.Chunk = importlib.import_module('hipporag.utils.misc_utils').Chunk
        self.QuerySolution = importlib.import_module('hipporag.utils.misc_utils').QuerySolution
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.target_book_id = target_book_id
        self.allowed_book_ids = set(allowed_book_ids)
        self.corpus_identity = None
        self._passage_metadata = {}
        self.retrieval_trace = []
        self.config = BaseConfig(llm_name=LLM_MODEL, llm_base_url=BASE_URL,
            embedding_model_name=EMBEDDING_MODEL, embedding_base_url=BASE_URL,
            embedding_provider='openai', llm_api_key_env='SILICONFLOW_API_KEY',
            embedding_api_key_env='SILICONFLOW_API_KEY', seed=42, temperature=0,
            response_format={'type': 'json_object'}, max_new_tokens=2048, openie_max_workers=4,
            openie_ner_max_tokens=2048, openie_triple_max_tokens=4096,
            embedding_max_seq_len=None, vector_store_type='parquet')
        audit = self.save_dir / 'native_calls.jsonl'
        self.extraction_llm = AuditedLLM(self.config, request_executor, audit)
        self.qa_llm = AuditedLLM(self.config, request_executor, audit, role='qa_llm')
        self.embedding_model = AuditedEmbedding(self.config, request_executor, audit)
        self.bridge_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        index_identity = _digest(dict(bridge=BRIDGE_VERSION, bridge_source=self.bridge_source_sha256, upstream=SOURCE_PIN,
            target_book_id=target_book_id, base_url=BASE_URL, llm=LLM_MODEL,
            embedding=EMBEDDING_MODEL, embedding_normalization='l2',
            openie_response_format=self.config.response_format,
            query_instruction_mode='ignored', request_policy='seed42_temp0_thinkingfalse'))
        self.native = module.HippoRAG(global_config=self.config, save_dir=str(self.save_dir),
            extraction_llm=self.extraction_llm, qa_llm=self.qa_llm,
            embedding_model=self.embedding_model, index_identity=index_identity)

    def _check_errors(self):
        for component in (self.extraction_llm, self.qa_llm, self.embedding_model):
            component.check_error()

    def index(self, rows):
        self._check_errors()
        docs, identity = prepare_documents(rows, self.target_book_id, self.allowed_book_ids)
        path = self.save_dir / 'native_corpus_identity.json'
        if path.exists():
            if json.loads(path.read_text('utf-8')) != identity:
                raise ValueError('Native target corpus differs from persisted index')
        else:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(identity, handle, sort_keys=True, ensure_ascii=False)
        self.corpus_identity = identity
        self._passage_metadata = {_digest(d['text']): d['metadata'] for d in docs}
        chunks = [self.Chunk(content=d['text'], source_id=d['source_id'], metadata=d['metadata']) for d in docs]
        try:
            self.native.index(docs=chunks)
        finally:
            self._check_errors()
        return self.metadata()

    def _validate_solutions(self, solutions):
        if self.corpus_identity is None:
            raise RuntimeError('Bind the frozen target corpus with index before retrieval')
        allowed = set(self.corpus_identity['passage_digests'])
        for solution in solutions:
            if not isinstance(solution, self.QuerySolution) or solution.gold_docs is not None or solution.gold_answers is not None:
                raise ValueError('Native QA accepts retrieval solutions without reference answers')
            if len(solution.docs) != len(solution.doc_metadata or []):
                raise ValueError('Native retrieval source metadata is missing')
            for text, metadata in zip(solution.docs, solution.doc_metadata):
                key = _digest(text)
                if key not in allowed or metadata.get('book_id') != self.target_book_id:
                    raise ValueError('Native result contains foreign target evidence')
                expected = self._passage_metadata[key]
                if any(metadata.get(field) != expected[field] for field in expected):
                    raise ValueError('Native result source metadata differs from the frozen corpus')

    def retrieve(self, queries, top_k=None):
        self._check_errors()
        if self.corpus_identity is None:
            raise RuntimeError('Index the frozen target corpus before retrieval')
        if not isinstance(queries, list) or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('Native retrieval accepts question strings only')
        try:
            solutions = self.native.retrieve(queries=queries, num_to_retrieve=top_k)
        finally:
            self._check_errors()
        self._validate_solutions(solutions)
        for solution in solutions:
            self.retrieval_trace.append(dict(query_digest=_digest(solution.question),
                selected_passages=len(solution.docs), graph_seed_count=len(solution.graph_seeds or []),
                native_dpr_fallback=not bool(solution.graph_seeds),
                requested_top_k=self.config.retrieval_top_k if top_k is None else top_k,
                retrieval_variant='native_default' if top_k is None else 'explicit_top_k'))
        return solutions

    def rag_qa(self, queries):
        self._check_errors()
        if not isinstance(queries, list):
            raise ValueError('Native QA input must be a list')
        solutions = self.retrieve(queries) if queries and isinstance(queries[0], str) else queries
        self._validate_solutions(solutions)
        try:
            result = self.native.rag_qa(queries=solutions)
        finally:
            self._check_errors()
        self._validate_solutions(result[0])
        return result

    def metadata(self):
        return dict(method='HippoRAG 2 native index/retrieve/rag_qa', bridge_version=BRIDGE_VERSION,
            bridge_source_sha256=self.bridge_source_sha256,
            upstream_url=SOURCE_URL, upstream_commit=SOURCE_PIN, target_book_id=self.target_book_id,
            llm_model=LLM_MODEL, embedding_model=EMBEDDING_MODEL, provider=BASE_URL,
            native_retrieval_top_k=self.config.retrieval_top_k, native_qa_top_k=self.config.qa_top_k,
            generation_limits=dict(ner=2048, triples=4096, recognition_memory=512, qa=2048),
            openie_response_format=self.config.response_format,
            corpus_identity=self.corpus_identity, retrieval_trace=list(self.retrieval_trace),
            adaptation_disclosures=['Frozen Flash and BGE-M3 replace upstream experimental model choices',
                'Official SiliconFlow endpoint; temperature=0, seed=42, enable_thinking=false',
                'Target-book-only index; one pre-existing corpus evidence unit per native passage',
                'Identical passage strings coalesce natively; evidence identifiers are aggregated',
                'BGE-M3 1024-dimensional vectors are L2-normalized; no query instruction or silent truncation',
                'Native English prompts and default MuSiQue QA template are retained',
                'Official BaseConfig.response_format=json_object is enabled only for OpenIE NER/triples; native QA and recognition-memory formats are unchanged',
                'Native recognition-memory empty-fact DPR fallback is recorded',
                'Length finish reasons are retained for the upstream OpenIE JSON repair and QA accounting',
                'No reference answers or gold documents enter index, retrieve or QA',
                'A common external answer budget, if used, is a separate disclosed variant'])

    def close(self):
        self.native.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
