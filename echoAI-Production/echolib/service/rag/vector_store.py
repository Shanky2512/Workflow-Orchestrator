"""
Vector Store Implementation using FAISS for Traditional RAG

This module provides a vector-based document storage and retrieval system
using FAISS for efficient similarity search and OpenAI for embeddings.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import faiss
from openai import OpenAI

from echolib.types import Document

logger = logging.getLogger(__name__)


class VectorDocumentStore:
    """
    FAISS-based vector store for document embeddings.
    
    Uses OpenAI embeddings for semantic search capabilities.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: OpenAI embedding model name
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.embedding_model = embedding_model
        self.dimension = dimension
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Vector embeddings will not work.")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store documents and their metadata
        self._docs: Dict[str, Document] = {}
        self._doc_ids: List[str] = []  # Maps FAISS index position to document ID
        
        logger.info(f"Initialized VectorDocumentStore with {embedding_model}, dimension={dimension}")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding vector or None if client not initialized
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Cannot generate embeddings.")
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def put(self, doc: Document) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            doc: Document to add
            
        Returns:
            True if successful, False otherwise
        """
        # Create embedding from title + content
        text = f"{doc.title}\n{doc.content}"
        embedding = self._get_embedding(text)
        
        if embedding is None:
            logger.error(f"Failed to create embedding for document {doc.id}")
            return False
        
        # Add to FAISS index
        embedding_2d = embedding.reshape(1, -1)
        self.index.add(embedding_2d)
        
        # Store document and ID mapping
        self._docs[doc.id] = doc
        self._doc_ids.append(doc.id)
        
        logger.debug(f"Added document {doc.id} to vector store (total: {len(self._docs)})")
        return True
    
    def get(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        return self._docs.get(doc_id)
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Search for documents using semantic similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        if not self._docs:
            logger.warning("Vector store is empty")
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to create query embedding, falling back to keyword search")
            return self._keyword_search(query, top_k)
        
        # Search FAISS index
        query_2d = query_embedding.reshape(1, -1)
        k = min(top_k, len(self._docs))
        distances, indices = self.index.search(query_2d, k)
        
        # Retrieve documents
        results = []
        for idx in indices[0]:
            if idx < len(self._doc_ids):
                doc_id = self._doc_ids[idx]
                doc = self._docs.get(doc_id)
                if doc:
                    results.append(doc)
        
        logger.debug(f"Vector search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Fallback keyword-based search.
        
        Args:
            query: Query text
            top_k: Maximum results to return
            
        Returns:
            List of matching documents
        """
        q = query.lower()
        results = [
            d for d in self._docs.values()
            if q in d.title.lower() or q in d.content.lower()
        ]
        return results[:top_k]
    
    def clear(self):
        """Clear all documents from the store."""
        self.index.reset()
        self._docs.clear()
        self._doc_ids.clear()
        logger.info("Vector store cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_documents": len(self._docs),
            "index_size": self.index.ntotal,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "has_openai_client": self.client is not None
        }
