from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

_WORD_RE = re.compile(r"[\w\-\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "have",
    "has",
    "for",
    "are",
    "was",
    "were",
    "will",
    "shall",
    "should",
    "would",
    "could",
    "can",
    "may",
    "might",
    "into",
    "onto",
    "about",
    "without",
    "between",
    "这些",
    "我们",
    "你们",
    "以及",
    "但是",
    "因为",
    "所以",
    "就是",
    "如果",
    "可以",
    "不是",
    "已经",
    "通过",
    "一个",
    "一些",
    "主要",
    "需要",
    "针对",
    "相关",
    "利用",
    "提高",
    "实现",
}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    """Split plain text into overlapping chunks.

    The function is intentionally simple so that it works for both Chinese and
    English文本。它会按照句子/段落拆分，并在必要时增加重叠保证上下文连续性。
    """

    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\u3000", " ").strip()
    if not normalized:
        return []

    # Prefer splitting on sentence boundaries; fall back to paragraph/newlines.
    sentences: List[str] = []
    # First, try to split on common sentence terminators for both languages.
    for block in re.split(r"\n{2,}", normalized):
        block = block.strip()
        if not block:
            continue
        # Split by punctuation while preserving the delimiter.
        parts = re.split(r"(?<=[。！？.!?])\s+", block)
        sentences.extend(part.strip() for part in parts if part.strip())

    if not sentences:
        sentences = [line.strip() for line in normalized.splitlines() if line.strip()]

    if not sentences:
        return [normalized]

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if buffer and buffer_len + sent_len > chunk_size:
            chunks.append(" ".join(buffer).strip())
            if overlap > 0 and chunks[-1]:
                # Keep tail of previous chunk as overlap seed.
                overlap_text = chunks[-1][-overlap:]
                buffer = [overlap_text, sent]
                buffer_len = len(overlap_text) + sent_len
            else:
                buffer = [sent]
                buffer_len = sent_len
        else:
            buffer.append(sent)
            buffer_len += sent_len

    if buffer:
        chunks.append(" ".join(buffer).strip())

    # Remove potential duplicates and ensure non-empty strings.
    return [chunk for chunk in chunks if chunk]


def extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    """Extract lightweight keywords from a piece of text.

    We use a very small heuristic keyword extractor in order to keep the
    dependency surface minimal. It counts the frequency of alphabetic/Chinese
    tokens and returns the most common non-stop words.
    """

    if not text:
        return []

    tokens = [token.lower() for token in _WORD_RE.findall(text) if len(token) > 1]
    filtered = [tok for tok in tokens if tok not in _STOPWORDS and not tok.isdigit()]
    if not filtered:
        return []

    freq = Counter(filtered)
    keywords = [word for word, _ in freq.most_common(max_keywords)]
    return keywords


def build_chunk_payloads(
    chunks: Iterable[str],
    source: str,
    max_keywords: int = 6,
) -> List[dict]:
    """Construct chunk payloads enriched with keywords for storage."""

    payloads: List[dict] = []
    for idx, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk, max_keywords=max_keywords)
        payloads.append(
            {
                "text": chunk,
                "chunk_index": idx,
                "source": source,
                "keywords": keywords,
            }
        )
    return payloads
