from typing import List

from langchain_core.documents import Document


def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks: List[Document] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_text = text[start:end]
        if not chunk_text:
            continue
        chunks.append(
            Document(
                page_content=chunk_text,
                metadata={"start": start, "end": min(end, len(text))},
            )
        )
    return chunks
