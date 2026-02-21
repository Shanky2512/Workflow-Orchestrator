import json
from typing import Dict

from .base import BaseParser


class JsonParser(BaseParser):
    def parse(self, raw_bytes: bytes) -> Dict[str, str]:
        data = json.loads(raw_bytes.decode("utf-8"))
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return {"content": content}
