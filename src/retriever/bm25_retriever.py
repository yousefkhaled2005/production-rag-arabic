from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.preprocessor.arabic_cleaner import ArabicTextNormalizer


class ArabicBM25Retriever:
    """Lexical sparse retriever tuned for Arabic legal terminology."""

    def __init__(self):
        self.normalizer = ArabicTextNormalizer()
        self.corpus_chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None

    def index(self, chunks: List[Dict]):
        self.corpus_chunks = chunks
        self.tokenized_corpus = [
            self.normalizer.normalize(chunk["text"]).split()
            for chunk in chunks
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.bm25 or not self.corpus_chunks:
            return []

        clean_query = self.normalizer.normalize(query).split()
        scores = self.bm25.get_scores(clean_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "chunk_id": self.corpus_chunks[idx]["chunk_id"],
                "document_id": self.corpus_chunks[idx]["document_id"],
                "text": self.corpus_chunks[idx]["text"],
                "metadata": self.corpus_chunks[idx].get("metadata", {}),
                "score": float(scores[idx]),
                "rank": rank + 1
            })
        return results
