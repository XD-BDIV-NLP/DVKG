import unittest
from course_experiments.evaluation.graph_rag_v2 import EvidenceRetriever
from course_experiments.evaluation.k12_alias_compat import entity_names, compatible_worker_command


class K12AliasCompatibilityTests(unittest.TestCase):
    def test_null_alias_is_empty_without_changing_canonical_name(self):
        retriever=EvidenceRetriever([],{'entities':[{'id':'n','name':'内存','aliases':None}]})
        with self.assertRaises(TypeError):retriever.names('n')
        self.assertEqual(entity_names(retriever,'n'),['内存'])

    def test_valid_aliases_match_frozen_names_exactly(self):
        for entity in [{'id':'n','canonical_name':'缓存','aliases':['Cache','缓存']},
                       {'id':'n','name':'内存'}, {'id':'n','canonical_name':None,'aliases':['RAM']}]:
            r=EvidenceRetriever([],{'entities':[entity]})
            self.assertEqual(entity_names(r,'n'),r.names('n'))

    def test_null_alias_matches_explicit_empty_list_retrieval(self):
        row=dict(book_id='b',chunk_id='c',evidence_ids=['e'],text='内存 保存程序',evidence=[dict(id='e',text='内存 保存程序')])
        normal=EvidenceRetriever([row],{'entities':[{'id':'n','name':'内存','aliases':[],'evidence_ids':['e']}]})
        nullable=EvidenceRetriever([row],{'entities':[{'id':'n','name':'内存','aliases':None,'evidence_ids':['e']}]})
        nullable.names=lambda nid:entity_names(nullable,nid)
        self.assertEqual(nullable.retrieve('内存','graph'),normal.retrieve('内存','graph'))

    def test_only_os_k12_worker_commands_are_adapted(self):
        command=['python','-u','-m','course_experiments.evaluation.run_repaired_full','worker','--base','base','--matrix','matrix','--job','osAB_K12']
        self.assertEqual(compatible_worker_command(command,'shim.py'),['python','-u','shim.py','worker','--base','base','--matrix','matrix','--job','osAB_K12'])
        for name in ['compAB_K12','osAB_F']:
            other=command[:-1]+[name]
            self.assertEqual(compatible_worker_command(other,'shim.py'),other)
        self.assertEqual(compatible_worker_command(['python','--job','osAB_K12'],'shim.py'),['python','--job','osAB_K12'])


if __name__=='__main__':unittest.main()
