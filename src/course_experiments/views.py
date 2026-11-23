from __future__ import annotations

import argparse
import json
from pathlib import Path
from .contributions import read_contribution, write_immutable


def semantic_view(nodes, mapping, facts, evidence):
    entities=[]
    members_by_canonical={}
    for native,canonical in mapping.items():
        members_by_canonical.setdefault(canonical,[]).append(native)
    facts_by_node={}
    for fact in facts:
        content=fact['content']
        if content.get('kind')=='node_fact':
            facts_by_node.setdefault(content['node_id'],[]).append(fact)
    for canonical in sorted(set(mapping.values())):
        node=nodes[canonical]
        if node['entity_type'] in {'Book','Chunk','Evidence'}:continue
        members=sorted(members_by_canonical[canonical])
        types={nodes[k]['entity_type'] for k in members}
        specific=types-{'Concept'}
        entity_type=next(iter(specific)) if len(specific)==1 else node['entity_type']
        sources=set()
        definitions=set()
        definition_variants=[]
        type_provenance=[]
        identity_provenance=[]
        for key in members:
            row=nodes[key]
            identity_provenance.append(dict(native_node_id=key,canonical_name=row['name'],
                aliases=sorted(set(row.get('aliases',[]))),scope=row.get('scope',''),
                evidence_ids=row.get('evidence_ids',[]),source_state=row.get('source_state')))
            for field in ('evidence_ids','definition_evidence_ids','mention_evidence_ids'):
                sources.update(row.get(field,[]))
            if row.get('definition'):
                definitions.add(row['definition'])
            for variant in row.get('definitions',[]):
                if variant.get('value'):
                    definitions.add(variant['value'])
                sources.update(variant.get('evidence_ids',[]))
                definition_variants.append(dict(variant,native_node_id=key))
            type_provenance.append(dict(native_node_id=key,type=row['entity_type'],
                evidence_ids=row.get('evidence_ids',[])))
        for fact in facts_by_node.get(canonical,[]):
            content=fact['content']
            if content.get('node_id')==canonical:
                sources.update(fact['evidence_ids'])
                if content.get('field') in {'definition','definition_detail'}:
                    definitions.add(content['value'])
                    definition_variants.append(dict(value=content['value'],
                        conditions=content.get('conditions',''),evidence_ids=fact['evidence_ids'],
                        fact_id=fact['fact_id']))
        scopes=sorted({str(r['scope']) for r in identity_provenance if r['scope']})
        aliases=sorted({name for r in identity_provenance for name in [r['canonical_name'],*r['aliases']] if name!=node['name']})
        entities.append(dict(entity_id=canonical,canonical_name=node['name'],type=entity_type,
            aliases=aliases,scope=scopes[0] if len(scopes)==1 else '',scopes=scopes,
            identity_provenance=identity_provenance,identity_policy='native member names and explicit aliases; no inferred equivalence across scopes',
            definition=node.get('definition'),source_node_ids=sorted(sources),native_node_ids=members,
            merged_source_node_ids=sorted(sources),merged_raw_entity_ids=members,
            merged_definitions=sorted(definitions),definition_variants=definition_variants,
            type_provenance=type_provenance))
    names={r['entity_id']:r['canonical_name'] for r in entities}
    relations=[]
    for fact in facts:
        c=fact['content']
        if c.get('kind')!='relation' or c['head'] not in names or c['tail'] not in names:continue
        relations.append(dict(relation_id=fact['fact_id'],head_entity_id=c['head'],tail_entity_id=c['tail'],head=names[c['head']],relation=c['relation'],tail=names[c['tail']],conditions=c.get('conditions',''),evidence=[evidence[e] for e in fact['evidence_ids']],source_node_ids=fact['evidence_ids'],evidences=[evidence[e].get('text','') for e in fact['evidence_ids']],source_state=fact['source_state'],supports=fact['supports']))
    return dict(entities=entities,relations=relations)


def export_view(view, output):
    output=Path(output)/view['view_id']
    outputs={'fused.json':view,'entities.json':view['entities'],'relations.json':view['relations'],'evidence.json':view['evidence'],'conditions.json':{f['fact_id']:f['content'].get('conditions','') for f in view['facts']},'sources.json':dict(input_hashes=view['input_hashes'],active_books=view['active_books'],alignment_mapping=view['alignment_mapping'],alignment_audit=view['alignment_audit'],facts=view['facts'],potential_revisions=view['potential_revisions']),'summary.json':dict(view_id=view['view_id'],course_id=view['course_id'],scope=view['scope'],active_books=view['active_books'],entities=len(view['entities']),relations=len(view['relations']))}
    outputs['concept_kg.json'] = dict(entities=view['entities'],relations=view['relations'],view_id=view['view_id'])
    outputs['extractable_doc_nodes.json'] = [dict(row,source_node_id=eid,content=row.get('text',row.get('content',''))) for eid,row in sorted(view['evidence'].items())]
    for name,value in outputs.items():write_immutable(output/name,value)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description='Build immutable derived course fusion view (offline fixture alignment)')
    parser.add_argument('--g0',required=True)
    parser.add_argument('--contribution',action='append',default=[])
    parser.add_argument('--active-book',action='append',default=[])
    parser.add_argument('--alignment')
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args(argv)
    from .fusion import build_view
    fixture=json.loads(Path(args.alignment).read_text('utf-8-sig')) if args.alignment else None
    view=build_view(read_contribution(args.g0),[read_contribution(p) for p in args.contribution],args.active_book,alignment=fixture)
    print(export_view(view,args.output_root))

if __name__=='__main__':main()
