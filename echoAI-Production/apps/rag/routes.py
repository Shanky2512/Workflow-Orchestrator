from fastapi import APIRouter, Depends, HTTPException, Query
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import RAGService
from echolib.service.rag.graph_rag_service import GraphRAGService
from echolib.service.rag.hybrid_rag_service import HybridRAGService

import os
import tempfile
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from echolib.di import container
from echolib.security import user_context
from echolib.types import *

from echolib.services import RAGService
from echolib.service.rag.graph_rag_service import GraphRAGService
from echolib.service.rag.hybrid_rag_service import HybridRAGService
from echolib.service.rag.trad_rag import TraditionalRAGService

def svc() -> RAGService:
    return container.resolve('rag.service')

def graph_svc() -> GraphRAGService:
    return container.resolve('rag.graph_service')

def hybrid_svc() -> HybridRAGService:
    return container.resolve('rag.hybrid_service')


def traditional_svc() -> TraditionalRAGService:
    return container.resolve('rag.traditional_service')

router = APIRouter(prefix='/rag', tags=['RAGApi'])

@router.post('/index')
async def index(docs: list[Document]):
    """
    Index documents into the traditional RAG vector store.
    
    Uses FAISS + OpenAI embeddings for semantic search.
    """
    return svc().indexDocs(docs).model_dump()

@router.get('/search')
async def search(
    q: str,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results to return")
):
    """
    Search documents using semantic similarity (vector search).
    
    Args:
        q: Query text
        top_k: Maximum number of results to return (1-50)
        
    Returns:
        ContextBundle with matching documents
    """
    return svc().queryIndex(q, {}, top_k=top_k).model_dump()

@router.get('/stats')
async def rag_stats():
    """
    Get traditional RAG vector store statistics.
    
    Returns:
        Statistics including document count, embedding model, and configuration
    """
    try:
        stats = svc().get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    

# ==================== TRADITIONAL RAG (FILE-BASED) ENDPOINTS ====================

@router.post('/traditional/load')
async def traditional_load(files: List[UploadFile] = File(..., description="PDF files to upload and index")):
    """
    Upload and load PDF documents into Traditional RAG.
    
    Uploads PDF documents, splits them into chunks, and creates FAISS embeddings
    for semantic search and question answering.
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        TraditionalRAGLoadResponse with loading summary
    """
    temp_dir = None
    try:
        # Create temporary directory to store uploaded files
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        
        saved_files = []
        for file in files:
            if not file.filename:
                continue
            # Validate file type (PDF only)
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Got: {file.filename}"
                )
            
            # Save file to temp directory
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        # Load documents from temp directory
        result = traditional_svc().load_documents(temp_dir)
        
        # Update path in response to reflect uploaded files
        result['path'] = f"Uploaded {len(saved_files)} file(s)"
        
        return TraditionalRAGLoadResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document loading failed: {str(e)}")
    finally:
        # Note: We don't delete temp_dir immediately as FAISS may need the files
        # In production, implement a cleanup strategy
        pass


@router.post('/traditional/query')
async def traditional_query(request: TraditionalRAGQueryRequest):
    """
    Generate an answer to a query using loaded documents.
    
    Uses the indexed documents to retrieve relevant context and generate
    a detailed answer using Azure OpenAI.
    
    Args:
        request: TraditionalRAGQueryRequest with query and optional custom prompt
        
    Returns:
        TraditionalRAGQueryResponse with answer and sources
    """
    try:
        result = traditional_svc().generate_response(
            query=request.query,
        )
        return TraditionalRAGQueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get('/traditional/stats')
async def traditional_stats():
    """
    Get Traditional RAG service statistics.
    
    Returns:
        TraditionalRAGStats with service configuration and document counts
    """
    try:
        result = traditional_svc().get_stats()
        return TraditionalRAGStats(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")



