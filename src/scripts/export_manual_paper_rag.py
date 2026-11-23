"""只读导出既有双视角RAG日志，按原始来源段落归并后离线评分。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course_experiments.manifest import digest
from scripts.manual_paper_metrics import rag_metrics, read, record

METHODS = ('dual', 'keyword', 'vector', 'hybrid', 'no_structure', 'no_mapping', 'no_semantic', 'no_relations')


def canonical_paragraphs(structure):
    book = structure['book_id'];mapping = {};boundaries = Counter();verified = 0
    for eid, row in structure['evidence'].items():
        source = row['source'];pid = source.get('source_id')
        if row['book_id'] != book or source['book_id'] != book or not isinstance(pid, str) or not pid:
            raise ValueError('Original paragraph identity or target book differs')
        mapping[eid] = book+'/paragraph/'+pid
        boundaries[source.get('paragraph_boundary_status', 'undisclosed')] += 1
        verified += source.get('natural_paragraph_boundary_verified') is True
    return mapping, dict(evidence_units=len(mapping), source_paragraphs=len(set(mapping.values())),
                         boundary_status_counts=dict(boundaries),
                         natural_boundary_verified_evidence_units=verified,
                         identity='target book + original source_id; evidence slices share paragraph identity')


def paragraph_ids(eids, mapping):
    if not isinstance(eids, list) or any(not isinstance(e, str) or e not in mapping for e in eids):
        raise ValueError('Unknown or foreign original evidence identity')
    return list(dict.fromkeys(mapping[e] for e in eids))


def normalize_prediction(question, saved, retrieval, mapping, result_id):
    if saved is None:
        return None
    if (saved.get('result_id') != result_id or saved.get('question_id') != question['question_id']
            or saved.get('target_book_id') != question['target_book_id']
            or saved.get('checkpoint_digest') != digest({k:v for k,v in saved.items() if k!='checkpoint_digest'})):
        raise ValueError('Question checkpoint identity or digest differs')
    status = saved.get('status')
    if status not in ('completed', 'failed', 'pending'):
        raise ValueError('Unsupported explicit question status')
    row = dict(question_id=question['question_id'], target_book_id=question['target_book_id'], status=status)
    if status == 'completed':
        if not retrieval or 'selected_evidence_ids' not in retrieval:
            raise ValueError('Explicit pre-packing retrieval order required; citations/context cannot substitute')
        if retrieval.get('result_id') != result_id:
            raise ValueError('Retrieval result identity differs')
        answer = saved['candidate']['answer']
        if not isinstance(answer, str):
            raise ValueError('Completed candidate answer required')
        row.update(answer=answer, retrieved_paragraph_ids=paragraph_ids(retrieval['selected_evidence_ids'], mapping))
    return row


def checked(source):
    if record(source['path']) != source:
        raise ValueError('Frozen source SHA differs: '+source['path'])
    return read(source['path'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def export_directory(base, output):
    base = Path(base);output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = read(base/'assets/result.json');checked(assets['contract'])
    datasets = read(base/'datasets/index.json')['books']
    mapping = {};boundary = {};questions = {}
    for book, source in assets['structures'].items():
        structure = checked(source)
        if structure['book_id'] != book:
            raise ValueError('Structure target differs')
        mapping[book], boundary[book] = canonical_paragraphs(structure)
        dataset = checked(datasets[book]['confirmation'])
        questions[book] = dataset['questions']
        if any(q['target_book_id'] != book for q in questions[book]):
            raise ValueError('Frozen question target differs')
    groups = {};errors = [];sources = []
    for cohort, source in assets['cohorts'].items():
        graph = checked(source['graph']);del graph
        course = checked(source['construction'])
        course_id = read(Path(source['graph']['path']).parent/'summary.json')['course_id']
        del course
        gold = []
        for book in source['active_books']:
            for q in questions[book]:
                gold.append(dict(question_id=q['question_id'], course_id=course_id, target_book_id=book,
                                 answer=q['answer'], paragraph_ids=paragraph_ids(q['evidence_ids'], mapping[book])))
        for method in METHODS:
            key = cohort+'__'+method;predictions = [];provenance = []
            try:
                for book in source['active_books']:
                    job = dict(name=cohort+'__'+book+'__'+method, cohort=cohort, target_book_id=book, method=method)
                    folder = base/'evaluations'/job['name'];contract_path = folder/'contract.json'
                    if not contract_path.exists():
                        continue
                    contract = read(contract_path);result_id = contract['result_id']
                    if (contract['job'] != job or contract['reference'] != datasets[book]['confirmation']
                            or contract['asset_contract'] != assets['contract']
                            or result_id != digest({k:v for k,v in contract.items() if k!='result_id'})):
                        raise ValueError('Job, question pool, assets or result contract differs')
                    expected = {q['question_id'] for q in questions[book]}
                    if {p.stem for p in (folder/'questions').glob('*.json')}-expected:
                        raise ValueError('Foreign question checkpoint')
                    provenance.append(record(contract_path))
                    for q in questions[book]:
                        answer_path = folder/'questions'/(q['question_id']+'.json')
                        retrieval_path = folder/'retrieval'/(q['question_id']+'.json')
                        saved = read(answer_path) if answer_path.exists() else None
                        retrieval = read(retrieval_path) if retrieval_path.exists() else None
                        prediction = normalize_prediction(q, saved, retrieval, mapping[book], result_id)
                        if prediction is not None:
                            predictions.append(prediction)
                            provenance.append(record(answer_path))
                            if retrieval is not None:
                                provenance.append(record(retrieval_path))
                metric = rag_metrics(gold, predictions)
                group = output/key
                write(group/'gold.json', dict(questions=gold))
                group.mkdir(exist_ok=True)
                with (group/'predictions.jsonl').open('x', encoding='utf-8') as stream:
                    for row in predictions:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                write(group/'metrics.json', metric)
                write(group/'provenance.json', dict(files=provenance, graph=source['graph']))
                groups[key] = dict(cohort=cohort, method=method, status=metric['status'], questions=metric['questions'],
                                   completed=metric['completed'], failed=metric['failed'], pending=metric['pending'],
                                   metrics=metric['course_macro'], path=str(group/'metrics.json'))
                sources.append(dict(cohort=cohort, method=method, gold=gold, predictions=predictions))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(dict(group=key, error_type=type(exc).__name__, reason=str(exc)))
    aggregates = {}
    for method in METHODS:
        for co in ('compAB', 'compABCD'):
            selected = [x for x in sources if x['method']==method and x['cohort'] in (co, 'osAB', 'netAB')]
            if len(selected) != 3:
                continue
            gold = [q for x in selected for q in x['gold']]
            predictions = [q for x in selected for q in x['predictions']]
            key = co+'__'+method;metric = rag_metrics(gold, predictions)
            write(output/'course_macro'/(key+'.json'), metric)
            aggregates[key] = dict(questions=metric['questions'], status=metric['status'], metrics=metric['course_macro'])
        ab = next((x for x in sources if x['method']==method and x['cohort']=='compAB'), None)
        abcd = next((x for x in sources if x['method']==method and x['cohort']=='compABCD'), None)
        if ab and abcd:
            fixed = {q['question_id'] for q in ab['gold']}
            paired = [q for q in abcd['gold'] if q['question_id'] in fixed]
            if digest(sorted(paired, key=lambda q:q['question_id'])) != digest(sorted(ab['gold'], key=lambda q:q['question_id'])):
                raise ValueError('AB/ABCD same-target question pairing differs')
            metric = rag_metrics(paired, [q for q in abcd['predictions'] if q['question_id'] in fixed])
            write(output/'paired_AB_questions'/('ABCD__'+method+'.json'), metric)
    manifest = dict(status='completed_legacy_log_export' if not errors and all(x['status']=='completed' for x in groups.values()) else 'partial',
                    groups=groups, course_macro=aggregates, errors=errors, model_calls=0, raw_text_exported=False,
                    input_assets=record(base/'assets/result.json'), datasets=record(base/'datasets/index.json'),
                    paragraph_boundary_audit=boundary, human_audited=False,
                    scope='existing target_book_dual_view_v1 frozen confirmation/regression logs; not current ECHA or native baselines',
                    disclosure='Recall uses original source_id paragraph units; OCR/PDF partition boundaries are not human-verified natural paragraphs; machine references are not human gold')
    write(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_directory(args.base, args.output)
    print(json.dumps(dict(status=result['status'], output=str(args.output), groups=len(result['groups']),
                          blocked_groups=len(result['errors']), model_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
