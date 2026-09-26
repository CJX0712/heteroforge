"""路由层测试: HAAR 双门控、级联真跳过、E504/E505 守卫。"""

from __future__ import annotations

import numpy as np
import pytest

from heteroforge.core.config import EmbedConfig, GNNConfig, RoutingParams
from heteroforge.core.errors import HeteroForgeError
from heteroforge.embed.channel_a import ChannelAEmbedder
from heteroforge.gnn.channel_b import SGCBackend
from heteroforge.graph.homophily import measure_homophily
from heteroforge.router.haar import HAARRouter, assert_no_test_access, default_alpha_fn


@pytest.fixture()
def routed_pair(tiny_graph):
    out_a = ChannelAEmbedder(preferred="spectral_rw", dim=16, seed=2).fit_predict(tiny_graph)
    gnn = SGCBackend(seed=2).fit(
        tiny_graph, GNNConfig(seed=2, n_epochs=10), labels_mask=tiny_graph.train_mask
    )
    report = measure_homophily(tiny_graph.adjacency, tiny_graph.labels)
    return tiny_graph, report, out_a, gnn


class TestAlphaFn:
    def test_low_homophily_favors_channel_a(self):
        params = RoutingParams(alpha_base=0.5, w_homophily=1.0)
        assert default_alpha_fn(0.0, params) == pytest.approx(1.0)
        assert default_alpha_fn(1.0, params) == pytest.approx(0.0)
        assert default_alpha_fn(0.5, params) == pytest.approx(0.5)

    def test_nan_homophily_neutral(self):
        params = RoutingParams(alpha_base=0.5, w_homophily=1.0)
        assert default_alpha_fn(float("nan"), params) == pytest.approx(0.5)


class TestGuards:
    def test_e504_blocks_test_access(self):
        with pytest.raises(HeteroForgeError) as ei:
            assert_no_test_access("train_mask", "test_mask")
        assert ei.value.errcode == "E504"

    def test_e504_allows_train_val(self):
        assert assert_no_test_access("train_mask", "val_mask") is None

    def test_e505_fusion_mask_mismatch(self, routed_pair):
        graph, report, out_a, gnn = routed_pair
        full_b = gnn.predict(graph)
        decision = HAARRouter().route(graph, report, out_a, None, RoutingParams(mode="cascade"))
        broken = full_b
        broken.computed_mask = np.zeros(graph.num_nodes, dtype=bool)
        with pytest.raises(HeteroForgeError) as ei:
            decision.fuse(out_a.fusion_matrix(), broken.fusion_matrix(), broken.computed_mask)
        assert ei.value.errcode == "E505"


class TestCascade:
    def test_cascade_true_skip(self, routed_pair):
        graph, report, out_a, gnn = routed_pair
        decision = HAARRouter().route(
            graph, report, out_a, None,
            RoutingParams(mode="cascade", threshold=0.99, budget_ratio=0.3),
        )
        assert 0 < decision.routed_fraction <= 0.3 + 1e-9
        assert not (decision.routed_mask).all()

    def test_full_dual_routes_all(self, routed_pair):
        graph, report, out_a, _ = routed_pair
        decision = HAARRouter().route(graph, report, out_a, None, RoutingParams(mode="full_dual"))
        assert decision.routed_mask.all()
        assert (decision.reason == "FORCED_FULL_DUAL").all()

    def test_fuse_fidelity(self, routed_pair):
        graph, report, out_a, gnn = routed_pair
        # w_homophily=0 使 alpha_fn 恒等于 alpha_base, 便于断言端点语义。
        decision = HAARRouter().route(
            graph, report, out_a, None,
            RoutingParams(mode="cascade", alpha_base=0.0, w_homophily=0.0,
                          threshold=0.99, budget_ratio=0.5),
        )
        routed_idx = np.flatnonzero(decision.routed_mask)
        out_b = gnn.predict(graph, nodes=routed_idx)
        fused = decision.fuse(out_a.fusion_matrix(), out_b.fusion_matrix(), out_b.computed_mask)
        accepted = ~decision.routed_mask
        if accepted.any():
            assert np.abs(fused[accepted] - out_a.fusion_matrix()[accepted]).max() <= 1e-12
        if routed_idx.size:
            assert np.abs(fused[routed_idx] - out_b.fusion_matrix()[routed_idx]).max() <= 1e-9

    def test_decision_validate_rejects_bad_reason(self, routed_pair):
        graph, report, out_a, _ = routed_pair
        decision = HAARRouter().route(graph, report, out_a, None, RoutingParams())
        decision.reason[0] = "BOGUS_REASON"
        with pytest.raises(HeteroForgeError) as ei:
            decision.validate()
        assert ei.value.errcode == "E502"
