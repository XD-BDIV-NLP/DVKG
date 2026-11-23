"""已有单书三元组按原生名称后合并；不声称原生双书重抽取。"""
from __future__ import annotations

from collections import defaultdict
import unicodedata


BOOKS = ('book_25f718562685', 'book_49bb8d31ddb9')
VARIANT = 'posthoc_AB_exact_name_union_v1'


def name_native_graph(document):
    """将KGGen书内ID还原为原生名称，禁止按局部ID跨书误合并。"""
    graph = document['native_graph']
    labels = {}
    for node in graph['nodes']:
        identity, name = node['id'], node['name']
        if not isinstance(identity, str) or not isinstance(name, str) or not name.strip():
            raise ValueError('Native nodes require explicit IDs and names')
        if identity in labels and labels[identity] != name:
            raise ValueError('Native node ID has conflicting names')
        labels[identity] = name
    edges = []
    for edge in graph['edges']:
        if edge['source'] not in labels or edge['target'] not in labels:
            raise ValueError('Native edge endpoint has no verified name')
        edges.append(dict(edge, source=labels[edge['source']], target=labels[edge['target']]))
    return dict(native_graph=dict(edges=edges, directed=graph.get('directed', True)))


def merge_book_triples(book_triples):
    """只归一首尾空白和NFC；保留全部谓词及逐事实教材来源。"""
    if set(book_triples) != set(BOOKS):
        raise ValueError('Exactly the frozen computer-organization A/B books are required')
    provenance = defaultdict(set)
    entity_sets = {}
    source_counts = {}
    for book in BOOKS:
        entities = set()
        count = 0
        for triple in book_triples[book]:
            if (not isinstance(triple, (tuple, list)) or len(triple) != 3 or
                    any(not isinstance(v, str) or not v.strip() for v in triple)):
                raise ValueError('Native entity names and predicates must be nonempty strings')
            h, r, t = (unicodedata.normalize('NFC', v.strip()) for v in triple)
            provenance[(h, r, t)].add(book)
            entities.update((h, t))
            count += 1
        if not count:
            raise ValueError('Both source books must contain native triples')
        entity_sets[book] = entities
        source_counts[book] = count
    triples = sorted(provenance)
    return dict(variant=VARIANT, books=list(BOOKS), triples=triples,
        triple_sources=[dict(triple=list(t), books=sorted(provenance[t])) for t in triples],
        counts=dict(source_triples=source_counts, input_triples=sum(source_counts.values()),
            unique_triples=len(triples), duplicates_removed=sum(source_counts.values())-len(triples),
            entities=len(set.union(*entity_sets.values())), relations=len({r for h, r, t in triples}),
            source_entities={k: len(v) for k, v in entity_sets.items()},
            shared_entities=len(set.intersection(*entity_sets.values())),
            shared_triples=sum(len(v) == 2 for v in provenance.values())),
        disclosure='post-hoc A/B graph union by exact native entity name; no fuzzy aliases, semantic relation mapping or native joint re-extraction; homonyms can merge and aliases can remain split')
