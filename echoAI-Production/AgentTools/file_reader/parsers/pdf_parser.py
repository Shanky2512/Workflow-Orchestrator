from typing import Dict

from pypdf import PdfReader

from .base import BaseParser


class PdfParser(BaseParser):
    def parse(self, raw_bytes: bytes) -> Dict[str, str]:
        reader = PdfReader(raw_bytes)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        content = "\n\n".join(pages).strip()
        return {"content": content}
