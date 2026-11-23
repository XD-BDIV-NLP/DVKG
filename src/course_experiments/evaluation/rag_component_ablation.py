"""复用冻结覆盖目标，单独关闭路由token成本。"""
from __future__ import annotations


def without_token_budget(original_select):
    """保留原生区域互斥、覆盖效用和数量上限；按覆盖增益排序。"""
    def select(structure,coverage,query_weights,paragraph_tokens,*,budget,max_regions):
        if (set(paragraph_tokens)!=set(structure['evidence']) or
                any(type(n) is not int or n<0 for n in paragraph_tokens.values())):
            raise ValueError('Exact original paragraph token counts are required')
        route=original_select(structure,coverage,query_weights,
            {e:0 for e in paragraph_tokens},budget=0,max_regions=max_regions)
        route.update(policy='gain_only_without_routing_token_budget_v1',
            routing_token_budget_applied=False,original_routing_token_budget=budget,
            tokens=sum(paragraph_tokens[e] for e in route['evidence_ids']),
            greedy_steps=[{k:v for k,v in step.items() if k!='tokens'} for step in route['greedy_steps']],
            token_count_source='actual_frozen_tokenizer_cost_reported_but_not_constrained')
        return route
    return select
