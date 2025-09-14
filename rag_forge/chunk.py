"""Chunking strategies. Just functions — pick one and call it."""

from __future__ import annotations

import re


def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with overlap.

    Dead simple, fast, and surprisingly hard to beat for well-structured docs.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap < chunk_size else end
    return chunks


def recursive_chunk(
    text: str, chunk_size: int = 512, overlap: int = 50, separators: list[str] | None = None
) -> list[str]:
    """Split on natural boundaries (paragraphs, sentences, words) recursively.

    Tries the biggest separator first. If a piece is still too big, falls back
    to the next separator. This tends to produce more coherent chunks than fixed.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # try each separator, biggest first
    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # if this single part is too big, recurse with next separator
                if len(part) > chunk_size:
                    remaining_seps = separators[separators.index(sep) + 1 :]
                    if remaining_seps:
                        chunks.extend(
                            recursive_chunk(part, chunk_size, overlap, remaining_seps)
                        )
                    else:
                        # last resort: hard split
                        chunks.extend(fixed_chunk(part, chunk_size, overlap))
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())

        if chunks:
            # add overlap between consecutive chunks
            if overlap > 0 and len(chunks) > 1:
                chunks = _add_overlap(chunks, overlap)
            return chunks

    # nothing worked, hard split
    return fixed_chunk(text, chunk_size, overlap)


def semantic_chunk(text: str, max_chunk_size: int = 1024, min_chunk_size: int = 100) -> list[str]:
    """Split on paragraph boundaries, then merge small paragraphs together.

    Not truly "semantic" (that would need embeddings), but respects document
    structure better than fixed chunking. Good enough for most cases — I found
    the embedding-based approach wasn't worth the extra latency in practice.
    """
    if not text.strip():
        return []

    # split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chunk_size:
            # big paragraph — flush current and split it
            if current.strip():
                chunks.append(current.strip())
                current = ""
            # use sentence splitting for oversized paragraphs
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= max_chunk_size:
                    buf = buf + " " + sent if buf else sent
                else:
                    if buf.strip():
                        chunks.append(buf.strip())
                    buf = sent
            if buf.strip():
                chunks.append(buf.strip())
        elif len(current) + len(para) + 2 <= max_chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    # merge tiny chunks with their neighbor
    merged = []
    for chunk in chunks:
        if merged and len(merged[-1]) < min_chunk_size:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return merged


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add trailing context from previous chunk to the start of each chunk."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # don't add overlap if it starts mid-word
        space_idx = prev_tail.find(" ")
        if space_idx != -1:
            prev_tail = prev_tail[space_idx + 1 :]
        result.append(prev_tail + " " + chunks[i] if prev_tail else chunks[i])
    return result


# map name -> function for the benchmark runner
CHUNKERS = {
    "fixed_512": lambda text: fixed_chunk(text, 512, 50),
    "fixed_256": lambda text: fixed_chunk(text, 256, 25),
    "recursive_512": lambda text: recursive_chunk(text, 512, 50),
    "semantic": lambda text: semantic_chunk(text, 1024, 100),
}
