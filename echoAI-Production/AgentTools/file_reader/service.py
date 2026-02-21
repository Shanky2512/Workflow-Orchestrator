import base64
import hashlib

from .models import FileReaderInput
from .stream_utils import stream_text
from .registry import PARSER_REGISTRY
from .vector_store import load_or_create, save
from .embeddings import split_into_chunks
from .summarizer import summarize_documents
from .csv_capability.csv_agent import CSVQueryAgent
from .csv_capability.csv_summarizer import CSVSummarizer

MAX_SIZE = 50 * 1024 * 1024


def is_tabular(name: str, mime: str) -> bool:
    return name.lower().endswith((".csv", ".xls", ".xlsx"))


def doc_id(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FileReaderService:
    def process(self, data: FileReaderInput):
        raw = base64.b64decode(data.content_base64)
        if len(raw) > MAX_SIZE:
            raise ValueError("File > 50MB")

        # CSV PATH
        if is_tabular(data.file_name, data.mime_type):
            if data.mode == "query":
                agent = CSVQueryAgent(stream=data.stream)
                exec_agent = agent.create_agent(raw, data.file_name)
                exec_agent.invoke({"input": data.query})
                text = agent.collector.text() if data.stream else exec_agent.invoke({"input": data.query})["output"]
                return stream_text(text) if data.stream else text

            if data.mode == "summarize":
                result = CSVSummarizer().summarize(raw, data.file_name, stream=data.stream)
                return stream_text(result) if data.stream else result

        # DOCUMENT PATH
        parser = PARSER_REGISTRY.get(data.mime_type)
        if not parser:
            raise ValueError(f"Unsupported mime_type: {data.mime_type}")
        parsed = parser.parse(raw)
        content = parsed["content"]

        store = load_or_create(doc_id(raw))
        chunks = split_into_chunks(content, data.chunk_size, data.chunk_overlap)
        store.add_documents(chunks)
        save(store, doc_id(raw))

        if data.mode == "query":
            docs = store.as_retriever().get_relevant_documents(data.query)
            answer = summarize_documents(docs)
            return stream_text(answer) if data.stream else answer

        if data.mode == "summarize":
            answer = summarize_documents(chunks)
            return stream_text(answer) if data.stream else answer

        return content
