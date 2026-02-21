from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class FileReaderInput(BaseModel):
    file_name: str
    mime_type: str
    content_base64: str

    mode: Literal["extract", "query", "summarize"] = "extract"
    query: Optional[str] = None
    stream: bool = False

    chunk_size: int = Field(default=1000, ge=200, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)


class FileReaderResult(BaseModel):
    file_name: str
    file_type: str
    metadata: Dict[str, Any]
    result: Any
