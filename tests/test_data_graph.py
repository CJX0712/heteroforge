"""data 层测试: 合成图、切分、同配性度量。"""

from __future__ import annotations

import numpy as np
import pytest

from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import SplitSpec
from heteroforge.data.split import split_edges, split_hash, split_nodes, train_adjacency_from_split
from heteroforge.data.synthetic import generate_graph, make_graph_data
from heteroforge.graph.homophily import measure_homophily, node_local_homophily


class TestSynthetic:
    def test_sbm_homophily_buckets(self):
        for target in (0.05, 0.5, 0.95):
            out = generate_graph(200, 5, target, seed=5, generator="sbm", tolerance=0.05)
            assert abs(out.homophily_measured - target) <= 0.06

    def test_make_graph_data_contract(self):
        g = make_graph_data(n_nodes=100, num_classes=4, homophily=0.6, seed=9)
        g.validate()
        assert g.features.shape == (100, 64)
        assert g.train_mask.sum() + g.val_mask.sum() + g.test_mask.sum() == 100

    def test_size_guard_e204(self):
        with pytest.raises(HeteroForgeError) as ei:
            generate_graph(99999, 5, 0.5, seed=1, limits={"max_nodes": 5000})
        assert ei.value.errcode == "E204"

    def test_unknown_generator_e102(self):
        with pytest.raises(HeteroForgeError) as ei:
            generate_graph(100, 5, 0.5, seed=1, generator="bogus")
        assert ei.value.errcode == "E102"

    def test_reproducible(self):
        a = make_graph_data(n_nodes=60, homophily=0.5, seed=77)
        b = make_graph_data(n_nodes=60, homophily=0.5, seed=77)
        assert a.node_hash() == b.node_hash()


class TestSplit:
    def test_node_split_disjoint_stratified(self, small_graph):
        tr, va, te = split_nodes(small_graph.labels, SplitSpec(), seed=1)
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
        assert tr.sum() + va.sum() + te.sum() == small_graph.num_nodes

    def test_split_hash_stable(self, small_graph):
        tr, va, te = split_nodes(small_graph.labels, SplitSpec(), seed=1)
        assert split_hash(tr, va, te) == split_hash(tr, va, te)

    def test_edge_split_disjoint_and_connected(self, small_graph):
        spec = SplitSpec()
        split = split_edges(small_graph.adjacency, spec, seed=2)
        split.assert_disjoint()
        train_adj = train_adjacency_from_split(small_graph.adjacency, split)
        assert train_adj.shape == small_graph.adjacency.shape

    def test_edge_split_leakage_e205(self, small_graph):
        spec = SplitSpec()
        split = split_edges(small_graph.adjacency, spec, seed=2)
        split.train_edges = np.vstack([split.train_edges, split.test_edges[:1]])
        with pytest.raises(HeteroForgeError) as ei:
            split.assert_disjoint()
        assert ei.value.errcode == "E205"


class TestHomophily:
    def test_triangle_doctest_value(self):
        import networkx as nx

        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        rep = measure_homophily(g, np.array([0, 1, 1]))
        assert abs(rep.edge_homophily - 1 / 3) < 1e-9

    def test_isolated_nan(self, small_graph):
        h_vec, degrees = node_local_homophily(small_graph.adjacency, small_graph.labels)
        assert np.isnan(h_vec[degrees == 0]).all()

    def test_assortativity_safe_on_degenerate(self):
        import networkx as nx

        g = nx.Graph()
        g.add_node(0)
        rep = measure_homophily(g, np.array([0]))
        assert np.isnan(rep.degree_assortativity)

    def test_perfect_homophily(self):
        import networkx as nx

        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        rep = measure_homophily(g, np.array([0, 0, 0]))
        assert rep.edge_homophily == 1.0
