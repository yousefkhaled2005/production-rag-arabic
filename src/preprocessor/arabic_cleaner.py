import re
from typing import Optional


class ArabicTextNormalizer:
    """
    High-performance Arabic text normalizer designed for NLP and IR pipelines.
    Handles diacritics (Tashkeel), Tatweel (Kashida), Hamza variations, and punctuation.
    """

    # Tashkeel (Arabic diacritics) regex
    TASHKEEL_REGEX = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    
    # Tatweel (Kashida)
    TATWEEL_REGEX = re.compile(r"\u0640")

    # Alef variations
    ALEF_REGEX = re.compile(r"[إأآا]")

    # Yaa and Alef Maksura
    YAA_REGEX = re.compile(r"[يى]")

    # Taa Marbuta and Haa
    TAA_MARBUTA_REGEX = re.compile(r"ة")

    # Extra whitespaces
    WHITESPACE_REGEX = re.compile(r"\s+")

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_yaa: bool = False,
        normalize_taa_marbuta: bool = False,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_yaa = normalize_yaa
        self.normalize_taa_marbuta = normalize_taa_marbuta

    def normalize(self, text: str) -> str:
        """Executes full normalization pipeline on raw Arabic string."""
        if not text or not isinstance(text, str):
            return ""

        # Remove Tashkeel
        if self.remove_diacritics:
            text = self.TASHKEEL_REGEX.sub("", text)

        # Remove Tatweel
        if self.remove_tatweel:
            text = self.TATWEEL_REGEX.sub("", text)

        # Normalize Alef variations (إ, أ, آ -> ا)
        if self.normalize_alef:
            text = self.ALEF_REGEX.sub("ا", text)

        # Normalize Yaa (ي, ى -> ي)
        if self.normalize_yaa:
            text = self.YAA_REGEX.sub("ي", text)

        # Normalize Taa Marbuta (ة -> ه)
        if self.normalize_taa_marbuta:
            text = self.TAA_MARBUTA_REGEX.sub("ه", text)

        # Clean multiple spaces
        text = self.WHITESPACE_REGEX.sub(" ", text).strip()
        return text
