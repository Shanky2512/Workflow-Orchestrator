import xml.etree.ElementTree as ET
from typing import Dict

from .base import BaseParser


class XmlParser(BaseParser):
    def parse(self, raw_bytes: bytes) -> Dict[str, str]:
        root = ET.fromstring(raw_bytes)
        content = ET.tostring(root, encoding="unicode")
        return {"content": content}
