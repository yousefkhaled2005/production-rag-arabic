from typing import List, Dict
import re


class ArabicSemanticChunker:
    """
    Chunks Arabic legal texts respecting paragraph and sentence boundaries
    to preserve judicial context.
    """

    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[\.،؛\n!؟])\s+")

    def __init__(self, chunk_size_words: int = 250, overlap_words: int = 40):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk_document(self, text: str, doc_id: str, metadata: Dict = None) -> List[Dict]:
        sentences = self.SENTENCE_SPLIT_REGEX.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk_words: List[str] = []
        current_chunk_text = ""
        chunk_idx = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if len(current_chunk_words) + len(sentence_words) <= self.chunk_size_words:
                current_chunk_words.extend(sentence_words)
            else:
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunks.append({
                        "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                        "document_id": doc_id,
                        "text": chunk_text,
                        "metadata": metadata or {}
                    })
                    chunk_idx += 1
                    # Keep overlap from the end
                    current_chunk_words = current_chunk_words[-self.overlap_words:] + sentence_words
                else:
                    current_chunk_words = sentence_words

        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "document_id": doc_id,
                "text": chunk_text,
                "metadata": metadata or {}
            })

        return chunks
