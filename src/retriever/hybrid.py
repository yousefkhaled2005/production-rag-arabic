from typing import List, Dict
from collections import defaultdict


class ReciprocalRankFusion:
    r"""
    Reciprocal Rank Fusion (RRF) algorithm to fuse BM25 and Dense Retrieval rankings.
    Formula: RRF_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}
    """

    def __init__(self, k_constant: int = 60):
        self.k_constant = k_constant

    def fuse(self, dense_results: List[Dict], bm25_results: List[Dict], top_k: int = 5) -> List[Dict]:
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Process dense ranking
        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            chunk_map[cid] = doc
            rrf_scores[cid] += 1.0 / (self.k_constant + rank + 1)

        # Process lexical BM25 ranking
        for rank, doc in enumerate(bm25_results):
            cid = doc["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = doc
            rrf_scores[cid] += 1.0 / (self.k_constant + rank + 1)

        # Sort by final RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results = []
        for rank, cid in enumerate(sorted_cids[:top_k]):
            item = dict(chunk_map[cid])
            item["rrf_score"] = rrf_scores[cid]
            item["final_rank"] = rank + 1
            fused_results.append(item)

        return fused_results
