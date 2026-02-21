from typing import Generator


def stream_text(text: str, size: int = 20) -> Generator[str, None, None]:
    for i in range(0, len(text), size):
        yield text[i:i + size]
