"""Lossless binary edge view of the official K12 aggregated edge format."""
from copy import deepcopy


def model_schema_view(raw):
    """Validation view follows the official Exercise name-or-stem rule."""
    value = deepcopy(raw)
    for node in value['nodes']:
        if node.get('label') == 'Exercise' and not node.get('name'):
            stem = node.get('stem')
            if isinstance(stem, str) and stem.strip():
                node['name'] = stem
    return value


def flat_native_edges(raw):
    ids = {node['id'] for node in raw['nodes']}
    result = []
    for edge in raw['edges']:
        if edge.get('source') not in ids:
            raise ValueError('Native K12 edge source is absent')
        if 'target_name_to_ids' in edge:
            targets = edge['target_name_to_ids']
            if not isinstance(targets, list) or not targets:
                raise ValueError('Native K12 aggregated targets are empty')
            targets = [item['target'] for item in targets]
        else:
            targets = [edge.get('target')]
        for target in targets:
            if target not in ids:
                raise ValueError('Native K12 edge target is absent')
            value = deepcopy(edge)
            value.update(target=target, predicate=edge['type'])
            result.append(value)
    return result
