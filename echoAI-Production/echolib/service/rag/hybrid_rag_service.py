"""
Hybrid RAG Service

Combines traditional RAG (vector search with FAISS + OpenAI) with Graph RAG (entity relationships)
for enhanced retrieval capabilities.

This service bridges vector-based semantic search with knowledge graph traversal,
enabling hybrid search strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

from echolib.types import Document, ContextBundle, GraphQueryResult
from echolib.services import RAGService
from .vector_store import VectorDocumentStore
from .graph_rag_service import GraphDocumentStore, GraphRAGService

logger = logging.getLogger(__name__)


class HybridRAGService:
    """
    Hybrid RAG Service combining vector-based and graph-based retrieval.
    
    Strategies:
    - Traditional: Semantic vector search using FAISS + OpenAI embeddings
    - Graph: Knowledge graph traversal with entity relationships
    - Hybrid: Combines both for comprehensive results
    """
    
    def __init__(
        self,
        vector_store: VectorDocumentStore,
        graph_store: GraphDocumentStore
    ):
        """
        Initialize hybrid service with both stores.
        
        Args:
            vector_store: VectorDocumentStore for semantic search
            graph_store: GraphDocumentStore for graph-based search
        """
        self.traditional_service = RAGService(vector_store)
        self.graph_service = GraphRAGService(graph_store)
        self.vector_store = vector_store
        self.graph_store = graph_store
        logger.info("Hybrid RAG Service initialized with vector and graph stores")
        
    def index_documents(self, docs: List[Document], build_graph: bool = True) -> Dict[str, Any]:
        """
        Index documents into both traditional and graph stores.
        
        Args:
            docs: Documents to index
            build_graph: Whether to build graph structure (default: True)
            
        Returns:
            Combined indexing summary
        """
        # Index into traditional RAG
        trad_summary = self.traditional_service.indexDocs(docs)
        
        # Index into Graph RAG if enabled
        graph_summary = None
        if build_graph:
            graph_summary = self.graph_service.index_documents(docs)
            
        return {
            "traditional": {
                "count": trad_summary.count
            },
            "graph": {
                "indexed_documents": graph_summary.indexed_documents if graph_summary else 0,
                "extracted_entities": graph_summary.extracted_entities if graph_summary else 0,
                "extracted_relationships": graph_summary.extracted_relationships if graph_summary else 0
            } if build_graph else None,
            "total_documents": len(docs)
        }
        
    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        max_results: int = 10,
        traversal_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Search using specified strategy.
        
        Args:
            query: Search query
            strategy: "traditional", "graph", or "hybrid" (default)
            max_results: Maximum results to return
            traversal_depth: Graph traversal depth (for graph/hybrid)
            
        Returns:
            Unified search results with documents and metadata
        """
        if strategy == "traditional":
            return self._search_traditional(query, max_results)
        elif strategy == "graph":
            return self._search_graph(query, max_results, traversal_depth)
        elif strategy == "hybrid":
            return self._search_hybrid(query, max_results, traversal_depth)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'traditional', 'graph', or 'hybrid'")
            
    def _search_traditional(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using vector semantic search."""
        context = self.traditional_service.queryIndex(query, {}, top_k=max_results)
        
        return {
            "strategy": "traditional",
            "query": query,
            "documents": [doc.model_dump() for doc in context.documents[:max_results]],
            "total_results": len(context.documents),
            "metadata": {
                "method": "vector_semantic_search"
            }
        }
        
    def _search_graph(self, query: str, max_results: int, traversal_depth: int) -> Dict[str, Any]:
        """Search using graph traversal."""
        result = self.graph_service.query_graph(query, max_results, traversal_depth)
        
        return {
            "strategy": "graph",
            "query": query,
            "documents": [doc.model_dump() for doc in result.documents],
            "entities": [ent.model_dump() for ent in result.entities],
            "relationships": [rel.model_dump() for rel in result.relationships],
            "total_results": len(result.documents),
            "metadata": result.metadata
        }
        
    def _search_hybrid(self, query: str, max_results: int, traversal_depth: int) -> Dict[str, Any]:
        """
        Hybrid search combining vector and graph approaches.
        
        Strategy:
        1. Get results from both methods
        2. Merge and deduplicate by document ID
        3. Score based on presence in both results
        4. Include graph context (entities/relationships) for all documents
        """
        # Get vector search results
        trad_context = self.traditional_service.queryIndex(query, {}, top_k=max_results * 2)
        trad_docs = {doc.id: doc for doc in trad_context.documents}
        
        # Get graph results
        graph_result = self.graph_service.query_graph(query, max_results * 2, traversal_depth)
        graph_docs = {doc.id: doc for doc in graph_result.documents}
        
        # Merge results with scoring
        doc_scores = defaultdict(float)
        
        # Score from traditional search
        for i, doc in enumerate(trad_context.documents):
            doc_scores[doc.id] += (len(trad_context.documents) - i) / len(trad_context.documents)
            
        # Score from graph search
        for i, doc in enumerate(graph_result.documents):
            doc_scores[doc.id] += (len(graph_result.documents) - i) / len(graph_result.documents)
            
        # Bonus for documents appearing in both
        common_ids = set(trad_docs.keys()) & set(graph_docs.keys())
        for doc_id in common_ids:
            doc_scores[doc_id] *= 1.5  # Boost by 50%
            
        # Sort by score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Get top documents
        merged_docs = []
        for doc_id in sorted_doc_ids[:max_results]:
            # Prefer graph version (has more context)
            doc = graph_docs.get(doc_id) or trad_docs.get(doc_id)
            if doc:
                merged_docs.append(doc)
                
        return {
            "strategy": "hybrid",
            "query": query,
            "documents": [doc.model_dump() for doc in merged_docs],
            "entities": [ent.model_dump() for ent in graph_result.entities],
            "relationships": [rel.model_dump() for rel in graph_result.relationships],
            "total_results": len(merged_docs),
            "metadata": {
                "traditional_results": len(trad_docs),
                "graph_results": len(graph_docs),
                "common_results": len(common_ids),
                "method": "hybrid_retrieval"
            }
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from both systems."""
        graph_stats = self.graph_service.get_stats()
        
        # Get vector store statistics
        vector_stats = self.vector_store.stats()
        
        return {
            "vector_rag": vector_stats,
            "graph_rag": graph_stats.model_dump(),
            "sync_status": {
                "documents_in_both": min(
                    vector_stats["total_documents"], 
                    graph_stats.total_documents
                ),
                "in_sync": vector_stats["total_documents"] == graph_stats.total_documents
            }
        }
