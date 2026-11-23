"""既有原文复核修图的单条件受控接口；不改运行中的F、G0或T。"""
from __future__ import annotations
from ..manifest import digest
from .dual_view_rag import DualViewRetriever,asset
from .dvkg_development import CompletedMappingsRetriever
from . import state_dual_view as state
from . import run_frozen_dual_view as frozen

POLICY=dict(state.POLICY,version='source_reviewed_corrected_F_dual_v1',methods=['F_corrected+S+M'],
    state_graph='pre_existing_source_reviewed_corrected_export;raw_F_preserved',
    selection='fixed_existing_four_source_reviewed_graphs;no_confirmation_score_selection',
    prior_scores='923_old_questions_and85_supplement_are_separate;never_copy_scores',
    answer_prompt=frozen.old.ANSWER,judge_prompt=frozen.JUDGE)


class CorrectedFRetriever(CompletedMappingsRetriever):
    def __init__(self,graph,structures,mappings,unit_assets,concept_ids,concept_vectors,*,variant='occurrence_mapping_v2'):
        if variant not in ['unchanged_v1','occurrence_mapping_v2']:raise ValueError('Frozen development variant required')
        if mappings.get('concept_graph_id')!=digest(graph) or mappings.get('structure_ids')!={b:asset(s)['structure_id'] for b,s in structures.items()}:
            raise ValueError('Corrected F mapping or same S identity differs')
        self.semantic_state,self.variant='F_corrected',variant
        super().__init__(graph,structures,mappings,unit_assets,concept_ids,concept_vectors)

    def retrieve(self,question,target_book_id,query_vector,method=None):
        if method not in [None,'F_corrected+S+M']:raise ValueError('One frozen corrected F dual condition required')
        base=CompletedMappingsRetriever if self.variant=='occurrence_mapping_v2' else DualViewRetriever
        selected,trace=base.retrieve(self,question,target_book_id,query_vector,'dual')
        return selected,dict(trace,method='F_corrected+S+M',state='F_corrected',state_policy=POLICY['version'],
            mapping_graph_id=self.original_mappings['concept_graph_id'],mapping_structures=self.original_mappings['structure_ids'],
            mapping_id=self.original_mappings.get('mapping_id'),development_variant=self.variant,
            semantic_graph_used_as_answer_evidence=False)


def evaluate_question(engine,question,query_vector,provider,folder,identity,stage):
    result=state.evaluate_state_question(engine,question,query_vector,provider,folder,identity,stage)
    result['state_policy']=POLICY['version']
    return result
