from echolib.di import container
from echolib.services import RAGService
from echolib.service.rag.vector_store import VectorDocumentStore
from echolib.service.rag.graph_rag_service import GraphDocumentStore, GraphRAGService
from echolib.service.rag.hybrid_rag_service import HybridRAGService
from echolib.service.rag.trad_rag import TraditionalRAGService
from echolib.service.rag.session_rag_manager import SessionRAGManager

# Traditional RAG with Vector Store (FAISS + OpenAI embeddings)
_vector_store = VectorDocumentStore(
    embedding_model="text-embedding-3-small",
    dimension=1536
)
# container.register('rag.store', lambda: _vector_store)
# container.register('rag.service', lambda: RAGService(_vector_store))

# # Graph RAG
# _graph_store = GraphDocumentStore()
# container.register('rag.graph_store', lambda: _graph_store)
# container.register('rag.graph_service', lambda: GraphRAGService(_graph_store))

# # Hybrid RAG (combines vector + graph)
# container.register('rag.hybrid_service', lambda: HybridRAGService(_vector_store, _graph_store))

# # Traditional RAG (file-based with Azure OpenAI)
# _traditional_rag_service = TraditionalRAGService()
# container.register('rag.traditional_service', lambda: _traditional_rag_service)

# Session-scoped RAG (for app chat file uploads)


# _session_rag_manager = SessionRAGManager()
# container.register('rag.session_manager', lambda: _session_rag_manager)