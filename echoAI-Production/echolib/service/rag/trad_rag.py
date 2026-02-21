

"""
Traditional RAG Service

Document-based RAG using FAISS vector store with Azure OpenAI embeddings.
Supports loading documents from files/directories, chunking, indexing, and query answering.
"""

import logging
import os
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class TraditionalRAGService:
    """
    Traditional RAG Service using FAISS + Azure OpenAI.
    
    Provides document loading, chunking, embedding, and query answering capabilities.
    """
    
    DEFAULT_RAG_PROMPT = """
You are a professional assistant that answers users questions professionally and in extreme detail.

Question:
{question}

Context:
{context}

Follow these steps to answer:
1. Analyze the user's question and find all the relevant information from the context.
2. Cover all the relevant information in your answer, and provide the correct reasonings as per the context.
3. You have to answer all the questions DIRECTLY from the given context and NOT your own knowledge.
4. Answers should be based on what is given, and stated in the context. 
5. DO NOT deviate or modify the context; use all the information stated as it is.
6. If the context doesn't contain the relevant / DIRECT information required to answer the user's query, then respond: "I apologize, but I don't have a complete answer for this question based on the available information.

Provide a detailed analysis based on these instructions, focusing solely on the question asked.
"""
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        chat_deployment: str = "gpt4o",
        embedding_deployment: str = "text-embedding-ada-002",
        api_version: str = "2024-08-01-preview",
        temperature: float = 0.1,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        retrieval_k: int = 10
    ):
        """
        Initialize Traditional RAG Service.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)
            chat_deployment: Chat model deployment name
            embedding_deployment: Embedding model deployment name
            api_version: Azure OpenAI API version
            temperature: LLM temperature for response generation
            chunk_size: Document chunk size for splitting
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of chunks to retrieve for context
        """
        # self.azure_endpoint = ""I will put it later""  # Set your base url
        # self.api_key = ""I will put it later"" # Set your openai api key
        # self.chat_deployment = "gpt4o"
        # self.embedding_deployment = "text-embedding-ada-002"
        # self.api_version = "2024-08-01-preview"
        # self.temperature = 0.1
        # self.chunk_size = 800
        # self.chunk_overlap = 200
        # self.retrieval_k = 10
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.chat_deployment,
            temperature=self.temperature
        )
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=self.embedding_deployment,
            model=self.embedding_deployment,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            chunk_size=1
        )
        
        # FAISS store (initialized when documents are loaded)
        self.db: Optional[FAISS] = None
        self.retriever = None
        self._document_count = 0
        self._chunk_count = 0
        self._loaded_sources: List[str] = []
        
        logger.info("Traditional RAG Service initialized")
    
    def _load_file(self, file_path: str) -> List:
        """
        Load a single file with format-appropriate loader.

        Detects file type by extension and uses the corresponding
        LangChain document loader.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of loaded documents.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif ext == '.docx':
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        elif ext in ('.txt', '.md', '.csv'):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    def _load_documents(self, path: str) -> List:
        """
        Load documents from file or directory.

        Args:
            path: File path or directory path

        Returns:
            List of loaded documents
        """
        if os.path.isfile(path):
            return self._load_file(path)
        elif os.path.isdir(path):
            loader = DirectoryLoader(
                path,
                loader_cls=PyMuPDFLoader,
                use_multithreading=True,
                max_concurrency=128,
                show_progress=True,
                silent_errors=True
            )
        else:
            raise ValueError(f"Path does not exist: {path}")

        documents = loader.load()
        return documents
    
    def _split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def _format_docs(self, docs: List) -> str:
        """
        Format retrieved documents for context.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        formatted_docs = []
        for doc in docs:
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            content = doc.page_content
            formatted_docs.append(f"{source_name}\n{content}")
        return "\n\n".join(formatted_docs)
    
    def load_documents(self, path: str) -> Dict[str, Any]:
        """
        Load and index documents from a file or directory.
        
        Args:
            path: File path or directory path to load
            
        Returns:
            Summary of loaded documents
        """
        logger.info(f"Loading documents from: {path}")
        
        # Load documents
        documents = self._load_documents(path)
        self._document_count = len(documents)
        logger.info(f"Loaded {self._document_count} documents")
        
        # Adjust metadata to include only file name
        sources = set()
        for doc in documents:
            if 'source' in doc.metadata:
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                sources.add(doc.metadata['source'])
        self._loaded_sources = list(sources)
        
        # Split into chunks
        chunks = self._split_documents(documents)
        self._chunk_count = len(chunks)
        logger.info(f"Created {self._chunk_count} chunks")
        
        # Create FAISS index
        self.db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={"k": self.retrieval_k})
        
        logger.info("FAISS index created successfully")
        
        return {
            "status": "success",
            "documents_loaded": self._document_count,
            "chunks_created": self._chunk_count,
            "sources": self._loaded_sources,
            "path": path
        }
    
    def add_documents(self, path: str) -> Dict[str, Any]:
        """
        Load and merge additional documents into existing index using FAISS.merge_from().

        If no index exists yet, creates a new one. Otherwise merges the new
        documents into the existing FAISS index.

        Args:
            path: File path or directory path to load additional documents from.

        Returns:
            Dict with chunks_added count and updated stats.
        """
        logger.info(f"Adding documents from: {path}")

        documents = self._load_documents(path)
        new_doc_count = len(documents)
        logger.info(f"Loaded {new_doc_count} additional documents")

        # Adjust metadata to include only file name
        new_sources = set()
        for doc in documents:
            if 'source' in doc.metadata:
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])
                new_sources.add(doc.metadata['source'])

        # Split into chunks
        chunks = self._split_documents(documents)
        new_chunk_count = len(chunks)
        logger.info(f"Created {new_chunk_count} new chunks")

        # Create new FAISS index from the new chunks
        new_db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)

        if self.db is None:
            self.db = new_db
        else:
            self.db.merge_from(new_db)

        self.retriever = self.db.as_retriever(search_kwargs={"k": self.retrieval_k})

        # Update stats
        self._document_count += new_doc_count
        self._chunk_count += new_chunk_count
        self._loaded_sources = list(set(self._loaded_sources) | new_sources)

        logger.info("Documents merged into FAISS index successfully")

        return {
            "status": "success",
            "chunks_added": new_chunk_count,
            "documents_added": new_doc_count,
            "total_documents": self._document_count,
            "total_chunks": self._chunk_count,
            "sources": self._loaded_sources,
        }

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a query using the indexed documents.
        
        Args:
            query: User's question
            custom_prompt: Optional custom RAG prompt template
            
        Returns:
            Response with answer and sources
        """
        if self.retriever is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(query)
        
        # Format context
        context = self._format_docs(retrieved_docs)
        
        # Get sources
        sources = list(set(
            doc.metadata.get('source', 'Unknown') 
            for doc in retrieved_docs
        ))
        
        # Create prompt
        prompt_template =self.DEFAULT_RAG_PROMPT
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create chain
        chain = RunnableParallel(
            context=RunnablePassthrough(),
            question=RunnablePassthrough(),
        ) | prompt | self.llm | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": query,
        })
        
        logger.info("Response generated successfully")
        
        return {
            "query": query,
            "answer": response,
            "sources": sources,
            "chunks_used": len(retrieved_docs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Service statistics
        """
        return {
            "initialized": self.db is not None,
            "documents_loaded": self._document_count,
            "chunks_indexed": self._chunk_count,
            "sources": self._loaded_sources,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "retrieval_k": self.retrieval_k,
                "chat_deployment": self.chat_deployment,
                "embedding_deployment": self.embedding_deployment
            }
        }
    
    def save_index(self, path: str) -> Dict[str, Any]:
        """
        Save FAISS index to disk.
        
        Args:
            path: Directory path to save the index
            
        Returns:
            Save status
        """
        if self.db is None:
            raise ValueError("No index to save. Call load_documents() first.")
        
        self.db.save_local(path)
        logger.info(f"Index saved to: {path}")
        
        return {
            "status": "success",
            "path": path
        }
    
    def load_index(self, path: str) -> Dict[str, Any]:
        """
        Load FAISS index from disk.
        
        Args:
            path: Directory path containing the saved index
            
        Returns:
            Load status
        """
        self.db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.db.as_retriever(search_kwargs={"k": self.retrieval_k})
        logger.info(f"Index loaded from: {path}")
        
        return {
            "status": "success",
            "path": path
        }