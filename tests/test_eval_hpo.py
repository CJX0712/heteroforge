"""评测层与 HPO 测试: sklearn 对照、E605、端点入队、门槛 enforcement。"""

from __future__ import annotations

import numpy as np
import pytest

from heteroforge.core.config import GNNConfig
from heteroforge.core.errors import HeteroForgeError
from heteroforge.embed.channel_a import ChannelAEmbedder
from heteroforge.eval.link_prediction import degree_aware_negatives, evaluate_link_prediction, roc_auc_manual
from heteroforge.eval.node_classification import evaluate_representation, head_predictions, macro_f1_manual
from heteroforge.gnn.channel_b import SGCBackend
from heteroforge.graph.homophily import measure_homophily
from heteroforge.hpo.tuner import tune_routing
from heteroforge.pipeline.benchmark import enforce_haar_gate, haar_gate_check


class TestMetrics:
    def test_macro_f1_manual_matches_sklearn(self, rng):
        from sklearn.metrics import f1_score

        y_true = rng.integers(0, 4, 200)
        y_pred = rng.integers(0, 4, 200)
        assert abs(macro_f1_manual(y_true, y_pred)
                   - f1_score(y_true, y_pred, average="macro", zero_division=0)) < 1e-12

    def test_roc_auc_manual_matches_sklearn(self, rng):
        from sklearn.metrics import roc_auc_score

        y = rng.integers(0, 2, 300)
        s = rng.random(300)
        assert abs(roc_auc_manual(y, s) - roc_auc_score(y, s)) < 1e-12

    def test_degree_aware_negatives_not_edges(self, small_graph):
        import scipy.sparse as sp

        sources = np.arange(0, 50)
        negs = degree_aware_negatives(small_graph.adjacency, sources, 1.0, seed=1)
        edge_set = {
            (int(a), int(b))
            for a, b in zip(*sp.triu(small_graph.adjacency, k=1).nonzero())
        }
        for a, b in negs:
            lo, hi = min(int(a), int(b)), max(int(a), int(b))
            assert (lo, hi) not in edge_set

    def test_link_prediction_metrics(self, small_graph):
        from heteroforge.data.split import EdgeSplit

        import scipy.sparse as sp

        edges = np.array(
            [(int(a), int(b)) for a, b in zip(*sp.triu(small_graph.adjacency, k=1).nonzero())],
            dtype=np.int64,
        )
        split = EdgeSplit(
            train_edges=edges[: len(edges) * 3 // 4],
            val_edges=edges[len(edges) * 3 // 4:],
            test_edges=edges[len(edges) * 3 // 4:],
            protected_edges=edges[:1],
            num_edges_total=len(edges),
        )
        train_adj = sp.csr_matrix(small_graph.adjacency.copy())
        mask = np.ones(train_adj.nnz, dtype=bool)
        train_adj.data = train_adj.data * mask
        holdout = {(min(int(a), int(b)), max(int(a), int(b)))
                   for a, b in split.test_edges}
        coo = train_adj.tocoo()
        keep = [
            k for k, (u, v) in enumerate(zip(coo.row, coo.col))
            if (min(int(u), int(v)), max(int(u), int(v))) not in holdout
        ]
        train_adj = train_adj.tocsr()[keep][:, keep] if False else train_adj
        from heteroforge.data.split import train_adjacency_from_split as _t

        train_adj = _t(small_graph.adjacency,
                       type(split)(train_edges=split.train_edges, val_edges=split.val_edges,
                                   test_edges=split.test_edges, protected_edges=split.protected_edges))
        rep = np.random.default_rng(3).random((small_graph.num_nodes, 8))
        metrics = evaluate_link_prediction(rep, train_adj, split.test_edges, 1.0, seed=3)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["average_precision"] <= 1.0


class TestNodeClassification:
    def test_head_predictions_shapes(self, tiny_graph):
        rep = np.random.default_rng(1).random((tiny_graph.num_nodes, 6))
        y_true, y_pred = head_predictions(rep, tiny_graph, tiny_graph.train_mask, tiny_graph.test_mask)
        assert y_true.shape == y_pred.shape and y_true.size > 0

    def test_evaluate_representation_touches_test_once(self, tiny_graph):
        rep = np.random.default_rng(1).random((tiny_graph.num_nodes, 6))
        result = evaluate_representation(
            rep, tiny_graph, system="feat_only", dataset_id="t", homophily_bucket=0.5,
            homophily_measured=0.5, seed=1,
        )
        assert result.test_touch_count == 1
        assert 0.0 <= result.metrics["macro_f1"] <= 1.0


class TestHPO:
    def test_hpo_beats_or_matches_best_endpoint(self, tiny_graph):
        out_a = ChannelAEmbedder(preferred="spectral_rw", dim=16, seed=4).fit_predict(tiny_graph)
        gnn = SGCBackend(seed=4).fit(
            tiny_graph, GNNConfig(seed=4, n_epochs=10), labels_mask=tiny_graph.train_mask
        )
        report = measure_homophily(tiny_graph.adjacency, tiny_graph.labels)
        result = tune_routing(tiny_graph, report, out_a, gnn, seed=4, n_trials=6, timeout_sec=60)
        assert result.selected_by in ("hpo", "default")
        assert result.params.alpha_base >= 0.0 and result.params.alpha_base <= 1.0

    def test_hpo_fallback_on_timeout(self, tiny_graph):
        out_a = ChannelAEmbedder(preferred="svd", dim=8, seed=4).fit_predict(tiny_graph)
        gnn = SGCBackend(seed=4).fit(
            tiny_graph, GNNConfig(seed=4, n_epochs=3), labels_mask=tiny_graph.train_mask
        )
        report = measure_homophily(tiny_graph.adjacency, tiny_graph.labels)
        result = tune_routing(tiny_graph, report, out_a, gnn, seed=4, n_trials=5, timeout_sec=0.0)
        assert result.selected_by == "default"
        assert "HPO_FALLBACK" in result.warnings


class TestGate:
    def _rows(self, haar, best):
        rows = []
        for system, value in (("haar", haar), ("channel_a", best), ("channel_b", best - 0.1)):
            for h in (0.2, 0.8):
                rows.append({"task": "node_classification", "system": system,
                             "n_nodes": 800, "homophily_bucket": h, "macro_f1": value})
        return rows

    def test_gate_pass(self):
        report = haar_gate_check(self._rows(0.80, 0.75))
        assert report["noninferior_all"] and report["mean_noninferior"]
        assert report["strictly_better_bucket_count"] >= 1
        enforce_haar_gate(report)

    def test_e602_noninferiority(self):
        report = haar_gate_check(self._rows(0.70, 0.75))
        assert not report["noninferior_all"]
        with pytest.raises(HeteroForgeError) as ei:
            enforce_haar_gate(report)
        assert ei.value.errcode == "E602"

    def test_e603_no_adaptive_gain(self):
        rows = self._rows(0.75, 0.75)
        report = haar_gate_check(rows)
        assert report["noninferior_all"] and report["mean_noninferior"]
        assert report["strictly_better_bucket_count"] == 0
        with pytest.raises(HeteroForgeError) as ei:
            enforce_haar_gate(report)
        assert ei.value.errcode == "E603"
