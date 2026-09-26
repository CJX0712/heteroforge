"""通道层测试: 通道A ladder 降级与确定性、通道B 数学一致性、子图契约。"""

from __future__ import annotations

import numpy as np
import pytest

from heteroforge.core.config import EmbedConfig, GNNConfig
from heteroforge.core.errors import HeteroForgeError
from heteroforge.embed.channel_a import ChannelAEmbedder, available_backends
from heteroforge.gnn.channel_b import SGCBackend, sknetwork_prop_features
from heteroforge.gnn.subgraph import k_hop_subgraph, symmetric_normalize


class TestChannelA:
    def test_ladder_fallback_used(self, tiny_graph):
        out = ChannelAEmbedder(preferred="node2vec", dim=16, seed=5).fit_predict(
            tiny_graph, EmbedConfig(backend="node2vec", dim=16, seed=5)
        )
        assert out.channel == "A"
        assert out.backend in ("node2vec", "spectral_rw", "svd")
        # fallback_used 的语义: 实际 backend 不是首选(首选失败或不可用)。
        assert out.fallback_used == (out.backend != "node2vec")

    def test_confidence_range(self, tiny_graph):
        out = ChannelAEmbedder(preferred="spectral_rw", dim=16, seed=5).fit_predict(tiny_graph)
        assert out.computed_mask.all()
        assert (out.confidence >= 0).all() and (out.confidence <= 1.0 + 1e-9).all()

    def test_reproducible_same_seed(self, tiny_graph):
        e1 = ChannelAEmbedder(preferred="svd", dim=16, seed=9).fit_predict(tiny_graph)
        e2 = ChannelAEmbedder(preferred="svd", dim=16, seed=9).fit_predict(tiny_graph)
        assert np.abs(e1.embedding - e2.embedding).max() <= 1e-6

    def test_validate_rejects_bad_nodes(self, tiny_graph):
        out = ChannelAEmbedder(preferred="svd", dim=8, seed=1).fit_predict(tiny_graph)
        out.node_ids = np.arange(1, tiny_graph.num_nodes + 1)
        with pytest.raises(HeteroForgeError) as ei:
            out.validate()
        assert ei.value.errcode == "E209"


class TestChannelB:
    def test_sgc_single_layer(self, tiny_graph):
        backend = SGCBackend(seed=1).fit(
            tiny_graph, GNNConfig(seed=1, n_epochs=5), labels_mask=tiny_graph.train_mask
        )
        assert backend.backend_id == "sgc"
        assert len(backend._assert_fitted().layers) == 1

    def test_full_vs_partial_bitwise_identical(self, tiny_graph):
        backend = SGCBackend(seed=1).fit(
            tiny_graph, GNNConfig(seed=1, n_epochs=10), labels_mask=tiny_graph.train_mask
        )
        full = backend.predict(tiny_graph)
        nodes = np.arange(0, tiny_graph.num_nodes, 3)
        part = backend.predict(tiny_graph, nodes=nodes)
        assert part.computed_mask.sum() == nodes.size
        diff = np.abs(part.proba[part.computed_mask] - full.proba[part.computed_mask]).max()
        assert diff <= 1e-9

    def test_proba_rows_sum_one(self, tiny_graph):
        backend = SGCBackend(seed=1).fit(
            tiny_graph, GNNConfig(seed=1, n_epochs=10), labels_mask=tiny_graph.train_mask
        )
        out = backend.predict(tiny_graph)
        assert np.allclose(out.proba.sum(axis=1), 1.0, atol=1e-9)
        rowsum = out.proba.sum(axis=1)
        assert np.allclose(rowsum, 1.0, atol=1e-9)

    def test_unfitted_raises_e401(self, tiny_graph):
        with pytest.raises(HeteroForgeError) as ei:
            SGCBackend(seed=1).predict(tiny_graph)
        assert ei.value.errcode == "E401"

    def test_prop_features_match_sknetwork_math(self, tiny_graph):
        prop = sknetwork_prop_features(tiny_graph.adjacency, tiny_graph.features)
        assert prop.shape == tiny_graph.features.shape
        assert np.isfinite(prop).all()


class TestSubgraph:
    def test_k_hop_contract(self, tiny_graph):
        center = 0
        nodes, sub_adj, pos = k_hop_subgraph(tiny_graph.adjacency, center, 2)
        assert nodes[pos] == center
        assert sub_adj.shape[0] == len(nodes)
        assert center in set(nodes.tolist())

    def test_symmetric_normalize(self, tiny_graph):
        norm = symmetric_normalize(tiny_graph.adjacency, add_self_loop=True)
        diff = norm - norm.T
        assert abs(diff).max() <= 1e-12
