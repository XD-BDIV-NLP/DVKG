"""八本教材小论文的手动结构统计与RAG评分；不调用模型、不启动训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course_experiments.evaluation.rag import score_answer
from course_experiments.evaluation.topology import CONCEPT_TYPES


def structure_metrics(graph, path_sources=0):
    import networkx as nx
    if path_sources < 0:
        raise ValueError('path_sources must be nonnegative')
    entities = graph.get('entities', graph.get('nodes', []))
    ids = [e.get('entity_id', e.get('id')) for e in entities]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Distinct nonempty graph entity IDs required')
    selected = {key for key, e in zip(ids, entities)
                if e.get('type') in CONCEPT_TYPES and e.get('type_status') != 'native_untyped_candidate'}
    network = nx.Graph();network.add_nodes_from(sorted(selected))
    known = set(ids)
    for edge in graph.get('relations', graph.get('edges', [])):
        a = edge.get('head_entity_id', edge.get('source'))
        b = edge.get('tail_entity_id', edge.get('target'))
        if a not in known or b not in known:
            raise ValueError('Dangling graph endpoint')
        if a in selected and b in selected and a != b:
            network.add_edge(a, b)
    n, m = network.number_of_nodes(), network.number_of_edges()
    components = sorted(nx.connected_components(network), key=lambda c: (-len(c), sorted(c)))
    largest = network.subgraph(components[0]) if components else network
    count = largest.number_of_nodes();distance = None;mode = 'undefined_fewer_than_two_nodes'
    if count >= 2:
        if path_sources and path_sources < count:
            sources = random.Random(42).sample(sorted(largest), path_sources)
            distance = sum(sum(nx.single_source_shortest_path_length(largest, s).values())
                           for s in sources) / (len(sources) * (count - 1))
            mode = 'sampled_sources_seed42'
        else:
            distance = nx.average_shortest_path_length(largest);mode = 'exact'
    return dict(projection='concept_semantic_simple_undirected_no_self_loops',
                nodes=n, edges=m, excluded_nodes=len(entities)-n,
                average_degree=2*m/n if n else None,
                density=nx.density(network) if n else None,
                average_clustering=nx.average_clustering(network, count_zeros=True) if n else None,
                components=len(components), isolates=len(list(nx.isolates(network))),
                largest_component_nodes=count, largest_component_fraction=count/n if n else None,
                average_path_length_lcc=distance, path_length_mode=mode,
                path_sources=min(path_sources, count) if path_sources else count)


def _ids(values):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError('Explicit canonical paragraph ID list required')
    return list(dict.fromkeys(values))


def _mean(rows):
    return {key: sum(r[key] for r in rows)/len(rows) if rows else None
            for key in ['answer_f1', 'paragraph_recall_at_5']}


def rag_metrics(gold, predictions):
    questions = {q['question_id']: q for q in gold}
    if not gold or len(questions) != len(gold):
        raise ValueError('Nonempty fixed gold questions with unique IDs required')
    predicted = {}
    for row in predictions:
        key = row['question_id']
        if key not in questions or key in predicted:
            raise ValueError('Foreign or duplicate predicted question ID')
        predicted[key] = row
    rows = []
    for key, question in questions.items():
        answer = question['answer'];book = question['target_book_id'];course = question['course_id']
        support = set(_ids(question['paragraph_ids']))
        if not support or not isinstance(answer, str) or not answer.strip() or not book or not course:
            raise ValueError('Gold answer, course, target book and nonempty paragraph support required')
        prediction = predicted.get(key)
        status = 'pending' if prediction is None else prediction.get('status', 'completed')
        if status not in ['pending', 'failed', 'completed']:
            raise ValueError('Explicit completed/failed/pending prediction status required')
        f1 = recall = 0.
        if prediction is not None and prediction.get('target_book_id') != book:
            raise ValueError('Target textbook differs from fixed gold')
        if status == 'completed':
            if not isinstance(prediction.get('answer'), str) or 'retrieved_paragraph_ids' not in prediction:
                raise ValueError('Completed answer and ranked retrieval paragraphs required; citations are insufficient')
            ranked = _ids(prediction['retrieved_paragraph_ids'])[:5]
            f1 = score_answer(prediction['answer'], answer)['f1']
            recall = len(support.intersection(ranked))/len(support)
        rows.append(dict(question_id=key, target_book_id=book, course_id=course,
                         status=status, answer_f1=f1, paragraph_recall_at_5=recall))
    books = {}
    for book in sorted({r['target_book_id'] for r in rows}):
        subset = [r for r in rows if r['target_book_id']==book]
        if len({r['course_id'] for r in subset}) != 1:
            raise ValueError('Each textbook must belong to one course')
        books[book] = dict(course_id=subset[0]['course_id'], questions=len(subset), **_mean(subset))
    courses = {course: _mean([r for r in books.values() if r['course_id']==course])
               for course in sorted({r['course_id'] for r in rows})}
    pending = sum(r['status']=='pending' for r in rows)
    return dict(status='partial' if pending else 'completed', questions=len(rows),
                completed=sum(r['status']=='completed' for r in rows),
                failed=sum(r['status']=='failed' for r in rows), pending=pending,
                denominator='all_fixed_gold_questions; pending and failed contribute zero; pending remains partial',
                retrieval_unit='canonical original paragraph; deduplicate ranked IDs then take first five; before context packing',
                micro=_mean(rows), books=books, courses=courses,
                course_macro=_mean(list(courses.values())), rows=rows)


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    topology = commands.add_parser('topology')
    source = topology.add_mutually_exclusive_group(required=True)
    source.add_argument('--graph', type=Path);source.add_argument('--assets', type=Path)
    topology.add_argument('--path-sources', type=int, default=0,
                          help='0为精确最大连通分量路径长度；正数为显式采样估计')
    topology.add_argument('--output', type=Path, required=True)
    rag = commands.add_parser('rag')
    rag.add_argument('--gold', type=Path, required=True)
    rag.add_argument('--predictions', type=Path, required=True)
    rag.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'topology':
        sources = ({cohort: row['graph'] for cohort, row in read(args.assets)['cohorts'].items()}
                   if args.assets else {'graph': record(args.graph)})
        rows = {}
        for label, source in sources.items():
            if record(source['path'])['sha256'] != source['sha256']:
                raise ValueError('Frozen graph SHA differs: '+label)
            rows[label] = dict(source=source, **structure_metrics(read(source['path']), args.path_sources))
        result = dict(protocol='manual_paper_structure_v1', rows=rows, model_calls=0)
    else:
        gold = read(args.gold)['questions']
        predictions = [json.loads(line) for line in args.predictions.read_text('utf-8-sig').splitlines() if line.strip()]
        result = dict(protocol='manual_paper_F1_paragraphRecall5_v1',
                      gold=record(args.gold), predictions=record(args.predictions),
                      **rag_metrics(gold, predictions), model_calls=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), model_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
