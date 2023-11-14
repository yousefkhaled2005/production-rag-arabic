from src.preprocessor.arabic_cleaner import ArabicTextNormalizer
from src.preprocessor.chunker import ArabicSemanticChunker


def test_tashkeel_removal():
    normalizer = ArabicTextNormalizer(remove_diacritics=True)
    raw = "القَانُونُ المَدَنِيُّ"
    cleaned = normalizer.normalize(raw)
    assert cleaned == "القانون المدني"


def test_tatweel_removal():
    normalizer = ArabicTextNormalizer(remove_tatweel=True)
    raw = "عــــقــــد"
    cleaned = normalizer.normalize(raw)
    assert cleaned == "عقد"


def test_chunking():
    chunker = ArabicSemanticChunker(chunk_size_words=10, overlap_words=2)
    sample_text = "المادة الأولى من القانون تقضي بالحقوق العامة. المادة الثانية تنظم واجبات العامل وصاحب العمل."
    chunks = chunker.chunk_document(sample_text, doc_id="law_1")
    assert len(chunks) >= 1
    assert chunks[0]["document_id"] == "law_1"
