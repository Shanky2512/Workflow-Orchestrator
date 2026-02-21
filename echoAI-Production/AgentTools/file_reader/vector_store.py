import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ROOT = "vector_db_storage"


def load_or_create(doc_id: str):
    path = os.path.join(ROOT, doc_id)
    os.makedirs(path, exist_ok=True)
    emb = OpenAIEmbeddings()

    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    return FAISS.from_texts([], emb)


def save(store, doc_id: str):
    store.save_local(os.path.join(ROOT, doc_id))
