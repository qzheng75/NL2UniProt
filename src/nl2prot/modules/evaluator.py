from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree


class Evaluator:
    def __init__(self, metric, **kwargs):
        self.metric = metric
        if self.metric == "TopKAcc":
            self.ks = kwargs.get("ks", [10])
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented yet")

    def evaluate(
        self,
        des_embedding: np.ndarray,
        seq_embedding: np.ndarray,
        ground_truth: np.ndarray,
    ) -> dict[str, float]:
        if self.metric == "TopKAcc":
            return {
                f"Top{k}Acc": self.topk_accuracy(
                    des_embedding, seq_embedding, ground_truth, k
                )
                for k in self.ks
            }
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented yet")

    def __find_k_nearest_neighbors(
        self, des_embedding: np.ndarray, seq_embedding: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tree = KDTree(seq_embedding)
        distance, indices = tree.query(des_embedding, k)
        return distance, indices

    def topk_accuracy(
        self,
        des_embedding: np.ndarray,
        seq_embedding: np.ndarray,
        ground_truth: np.ndarray,
        k: int,
    ) -> float:
        _, indices = self.__find_k_nearest_neighbors(des_embedding, seq_embedding, k)
        correct = np.any(indices == ground_truth[:, np.newaxis], axis=1)
        return np.mean(correct)