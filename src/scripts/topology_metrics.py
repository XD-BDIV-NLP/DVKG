"""保留孤立点的全图效率、有限距离与经典边Forman曲率。"""
from __future__ import annotations

from collections import Counter
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def metrics(graph):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError('Simple undirected loop-free graph required')
    n,m=len(graph),graph.number_of_edges()
    finite_pairs=0;distance_sum=0;inverse_sum=0.;lcc_distance=None
    components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),sorted(c)))
    for component_index,component in enumerate(components):
        size=len(component)
        if size<2:continue
        sub=graph.subgraph(component)
        matrix=nx.to_scipy_sparse_array(sub,nodelist=sorted(sub),dtype=np.float64,format='csr')
        matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
        component_sum=0;component_pairs=0
        for start in range(0,size,128):
            distances=shortest_path(matrix,directed=False,unweighted=True,method='D',
                indices=np.arange(start,min(size,start+128),dtype=np.int32))
            assert np.isfinite(distances).all()
            positive=distances>0
            component_pairs+=int(positive.sum());component_sum+=int(distances.sum())
            inverse_sum+=float(np.reciprocal(distances,out=np.zeros_like(distances),where=positive).sum())
        assert component_pairs==size*(size-1)
        finite_pairs+=component_pairs;distance_sum+=component_sum
        if component_index==0:lcc_distance=component_sum/component_pairs
    degrees=dict(graph.degree())
    curvature=Counter(4-degrees[u]-degrees[v] for u,v in graph.edges())
    curvature_sum=sum(k*count for k,count in curvature.items())
    degree_square_sum=sum(v*v for v in degrees.values())
    assert curvature_sum==4*m-degree_square_sum
    possible=n*(n-1)
    return dict(concept_count=n,simple_edges=m,components=len(components),isolates=nx.number_of_isolates(graph),
        global_efficiency=inverse_sum/possible if possible else 0.,
        finite_pair_mean_distance=distance_sum/finite_pairs if finite_pairs else None,
        mean_forman_edge_curvature=curvature_sum/m if m else None,
        average_path_length_lcc=lcc_distance,finite_ordered_pairs=finite_pairs,
        possible_ordered_pairs=possible,reachable_pair_fraction=finite_pairs/possible if possible else None,
        finite_distance_sum=distance_sum,inverse_distance_sum=inverse_sum,
        forman_edge_curvature_sum=curvature_sum,degree_square_sum=degree_square_sum,
        forman_edge_curvature_histogram={str(k):v for k,v in sorted(curvature.items())},
        exact=True,all_components_included=True,isolates_retained=True)
