"""
Graph RAG Service

Implements knowledge graph-based retrieval augmented generation.
Extracts entities and relationships from documents to build a semantic graph.

Architecture:
- GraphDocumentStore: Manages graph structure (entities, relationships)
- Entity Extraction: Uses LLM to identify key entities in documents
- Relationship Extraction: Uses LLM to identify semantic connections
- Graph Query: Traverses graph to find relevant context
- Hybrid Retrieval: Combines traditional vector search with graph traversal
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from echolib.types import (
    Document, 
    Entity, 
    Relationship, 
    GraphDocument, 
    GraphQueryResult,
    GraphStats,
    GraphIndexSummary
)
from echolib.utils import new_id

logger = logging.getLogger(__name__)


class GraphDocumentStore:
    """
    In-memory graph store for documents, entities, and relationships.
    
    Maintains:
    - Documents: Original content
    - Entities: Extracted concepts/entities
    - Relationships: Connections between entities
    - Indexes: For fast lookups
    """
    
    def __init__(self):
        # Core storage
        self._documents: Dict[str, Document] = {}
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        
        # Indexes for fast lookups
        self._entity_by_name: Dict[str, List[str]] = defaultdict(list)  # name -> entity_ids
        self._entity_by_type: Dict[str, List[str]] = defaultdict(list)  # type -> entity_ids
        self._relationships_by_entity: Dict[str, List[str]] = defaultdict(list)  # entity_id -> relationship_ids
        self._entities_by_doc: Dict[str, List[str]] = defaultdict(list)  # doc_id -> entity_ids
        
    def add_document(self, doc: Document) -> None:
        """Add a document to the store."""
        self._documents[doc.id] = doc
        
    def add_entity(self, entity: Entity) -> None:
        """Add an entity and update indexes."""
        self._entities[entity.id] = entity
        
        # Update indexes
        self._entity_by_name[entity.name.lower()].append(entity.id)
        self._entity_by_type[entity.type].append(entity.id)
        
        for doc_id in entity.source_doc_ids:
            self._entities_by_doc[doc_id].append(entity.id)
            
    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship and update indexes."""
        self._relationships[rel.id] = rel
        
        # Index by both source and target entities
        self._relationships_by_entity[rel.source_entity_id].append(rel.id)
        self._relationships_by_entity[rel.target_entity_id].append(rel.id)
        
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
        
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
        
    def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """Get relationship by ID."""
        return self._relationships.get(rel_id)
        
    def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[Entity]:
        """Find entities by name (exact or fuzzy match)."""
        name_lower = name.lower()
        
        if not fuzzy:
            entity_ids = self._entity_by_name.get(name_lower, [])
            return [self._entities[eid] for eid in entity_ids]
        
        # Fuzzy matching: find entities whose names contain the query
        matches = []
        for entity_name, entity_ids in self._entity_by_name.items():
            if name_lower in entity_name or entity_name in name_lower:
                matches.extend([self._entities[eid] for eid in entity_ids])
        return matches
        
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a given type."""
        entity_ids = self._entity_by_type.get(entity_type, [])
        return [self._entities[eid] for eid in entity_ids]
        
    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity."""
        rel_ids = self._relationships_by_entity.get(entity_id, [])
        return [self._relationships[rid] for rid in rel_ids]
        
    def get_connected_entities(self, entity_id: str, max_depth: int = 1) -> Set[str]:
        """
        Get all entities connected to the given entity up to max_depth.
        
        Uses BFS traversal.
        """
        connected = set()
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            connected.add(current_id)
            
            # Get relationships involving current entity
            for rel in self.get_entity_relationships(current_id):
                # Add connected entity
                if rel.source_entity_id == current_id:
                    neighbor_id = rel.target_entity_id
                else:
                    neighbor_id = rel.source_entity_id
                    
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
                    
        return connected
        
    def get_stats(self) -> GraphStats:
        """Get statistics about the graph."""
        entity_types = defaultdict(int)
        for entity in self._entities.values():
            entity_types[entity.type] += 1
            
        relationship_types = defaultdict(int)
        for rel in self._relationships.values():
            relationship_types[rel.relationship_type] += 1
            
        return GraphStats(
            total_documents=len(self._documents),
            total_entities=len(self._entities),
            total_relationships=len(self._relationships),
            entity_types=dict(entity_types),
            relationship_types=dict(relationship_types)
        )


class GraphRAGService:
    """
    Graph RAG Service - Knowledge graph-based retrieval.
    
    Features:
    - Entity extraction from documents using LLM
    - Relationship extraction using LLM
    - Graph-based query with path finding
    - Hybrid retrieval (vector + graph)
    """
    
    def __init__(self, store: GraphDocumentStore):
        self.store = store
        self._llm = None
        
    def _get_llm(self):
        """Get LLM instance for entity/relationship extraction."""
        if self._llm is None:
            try:
                from llm_manager import LLMManager
                self._llm = LLMManager.get_llm(temperature=0.1, max_tokens=2000)
                if self._llm:
                    logger.info("LLM initialized for Graph RAG entity/relationship extraction")
                else:
                    logger.warning("LLM not available - will use fallback extraction methods")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e} - using fallback methods")
                self._llm = None
        return self._llm
        
    def extract_entities(self, doc: Document) -> List[Entity]:
        """
        Extract entities from document using LLM.
        
        Uses few-shot prompting to identify:
        - People, organizations, locations
        - Concepts, technologies, products
        - Events, dates, metrics
        """
        llm = self._get_llm()
        if not llm:
            # Fallback: simple keyword extraction
            return self._extract_entities_fallback(doc)
            
        prompt = f"""Extract key entities from the following document.

Document Title: {doc.title}
Document Content: {doc.content}

Identify:
- People (PERSON)
- Organizations (ORGANIZATION)
- Locations (LOCATION)
- Concepts/Topics (CONCEPT)
- Technologies/Products (TECHNOLOGY)
- Events (EVENT)
- Dates/Times (DATE)

For each entity, provide:
1. Name (the entity's name)
2. Type (one of the types above)
3. Description (brief description from context)

Return as JSON array:
[
  {{"name": "Entity Name", "type": "TYPE", "description": "Brief description"}},
  ...
]

Extract at most 20 entities. Focus on the most important ones.
Return ONLY the JSON array, no markdown."""

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            entity_data = json.loads(content)
            
            # Convert to Entity objects
            entities = []
            for item in entity_data:
                entity = Entity(
                    id=new_id("ent_"),
                    name=item.get("name", "Unknown"),
                    type=item.get("type", "CONCEPT").upper(),
                    description=item.get("description"),
                    source_doc_ids=[doc.id],
                    metadata={"extraction_method": "llm"}
                )
                entities.append(entity)
                
            logger.info(f"Extracted {len(entities)} entities from document {doc.id}")
            return entities
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, using fallback")
            return self._extract_entities_fallback(doc)
            
    def _extract_entities_fallback(self, doc: Document) -> List[Entity]:
        """
        Fallback entity extraction using simple heuristics.
        
        Looks for:
        - Capitalized phrases (potential proper nouns)
        - Common entity patterns
        """
        entities = []
        content = doc.content
        
        # Simple pattern: Find capitalized words/phrases
        # Matches sequences like "John Smith", "New York", "Artificial Intelligence"
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, content)
        
        # Deduplicate and take top matches
        unique_matches = list(set(matches))[:10]
        
        for match in unique_matches:
            entity = Entity(
                id=new_id("ent_"),
                name=match,
                type="CONCEPT",  # Default type
                description=f"Extracted from {doc.title}",
                source_doc_ids=[doc.id],
                metadata={"extraction_method": "fallback"}
            )
            entities.append(entity)
            
        return entities
        
    def extract_relationships(self, doc: Document, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities using LLM.
        
        Identifies semantic connections like:
        - works_for, member_of, located_in
        - related_to, uses, produces
        - causes, precedes, depends_on
        """
        if len(entities) < 2:
            return []
            
        llm = self._get_llm()
        if not llm:
            # Fallback: create basic relationships
            return self._extract_relationships_fallback(doc, entities)
            
        # Limit entities to avoid token overflow
        entity_list = entities[:15]
        entity_names = [f"{e.name} ({e.type})" for e in entity_list]
        
        prompt = f"""Analyze relationships between these entities from a document.

Document: {doc.title}
Entities: {', '.join(entity_names)}

Identify meaningful relationships between these entities. For each relationship:
1. Source entity name
2. Target entity name  
3. Relationship type (e.g., works_for, related_to, part_of, uses, located_in, causes)
4. Brief description

Return as JSON array:
[
  {{"source": "Entity1", "target": "Entity2", "type": "relationship_type", "description": "Brief description"}},
  ...
]

Return at most 15 relationships. Focus on the most important connections.
Return ONLY the JSON array, no markdown."""

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            rel_data = json.loads(content)
            
            # Map entity names to IDs
            name_to_entity = {e.name: e for e in entity_list}
            
            # Convert to Relationship objects
            relationships = []
            for item in rel_data:
                source_name = item.get("source", "")
                target_name = item.get("target", "")
                
                source_entity = name_to_entity.get(source_name)
                target_entity = name_to_entity.get(target_name)
                
                if source_entity and target_entity:
                    rel = Relationship(
                        id=new_id("rel_"),
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relationship_type=item.get("type", "related_to").lower(),
                        description=item.get("description"),
                        source_doc_ids=[doc.id],
                        metadata={"extraction_method": "llm"}
                    )
                    relationships.append(rel)
                    
            logger.info(f"Extracted {len(relationships)} relationships from document {doc.id}")
            return relationships
            
        except Exception as e:
            logger.warning(f"LLM relationship extraction failed: {e}, using fallback")
            return self._extract_relationships_fallback(doc, entities)
            
    def _extract_relationships_fallback(self, doc: Document, entities: List[Entity]) -> List[Relationship]:
        """
        Fallback relationship extraction.
        
        Creates basic "related_to" relationships between co-occurring entities.
        """
        relationships = []
        
        # Create relationships between entities that co-occur in the same document
        for i, source in enumerate(entities[:10]):
            for target in entities[i+1:i+4]:  # Limit connections
                rel = Relationship(
                    id=new_id("rel_"),
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    relationship_type="related_to",
                    description=f"Co-occur in document {doc.title}",
                    strength=0.5,
                    source_doc_ids=[doc.id],
                    metadata={"extraction_method": "fallback"}
                )
                relationships.append(rel)
                
        return relationships
        
    def index_documents(self, docs: List[Document]) -> GraphIndexSummary:
        """
        Index documents into the knowledge graph.
        
        For each document:
        1. Extract entities
        2. Extract relationships
        3. Merge with existing graph
        4. Update indexes
        """
        start_time = time.time()
        
        total_entities = 0
        total_relationships = 0
        
        for doc in docs:
            # Add document
            self.store.add_document(doc)
            
            # Extract entities
            entities = self.extract_entities(doc)
            total_entities += len(entities)
            
            # Merge with existing entities (deduplicate by name)
            merged_entities = self._merge_entities(entities)
            
            # Add entities to store
            for entity in merged_entities:
                self.store.add_entity(entity)
                
            # Extract relationships
            relationships = self.extract_relationships(doc, merged_entities)
            total_relationships += len(relationships)
            
            # Add relationships to store
            for rel in relationships:
                self.store.add_relationship(rel)
                
        processing_time = (time.time() - start_time) * 1000
        stats = self.store.get_stats()
        
        return GraphIndexSummary(
            indexed_documents=len(docs),
            extracted_entities=total_entities,
            extracted_relationships=total_relationships,
            processing_time_ms=processing_time,
            stats=stats
        )
        
    def _merge_entities(self, new_entities: List[Entity]) -> List[Entity]:
        """
        Merge new entities with existing ones to avoid duplicates.
        
        If an entity with the same name exists, merge their properties.
        """
        merged = []
        
        for entity in new_entities:
            # Check if entity with same name exists
            existing = self.store.find_entities_by_name(entity.name, fuzzy=False)
            
            if existing:
                # Merge with existing entity
                existing_entity = existing[0]
                existing_entity.source_doc_ids.extend(entity.source_doc_ids)
                existing_entity.source_doc_ids = list(set(existing_entity.source_doc_ids))
                merged.append(existing_entity)
            else:
                # New entity
                merged.append(entity)
                
        return merged
        
    def query_graph(
        self, 
        query: str, 
        max_results: int = 10,
        traversal_depth: int = 2
    ) -> GraphQueryResult:
        """
        Query the knowledge graph.
        
        Strategy:
        1. Find entities matching the query
        2. Traverse graph to find connected entities (up to depth)
        3. Retrieve documents containing relevant entities
        4. Score and rank results
        """
        # Find seed entities from query
        seed_entities = self._find_query_entities(query)
        
        if not seed_entities:
            return GraphQueryResult(
                documents=[],
                entities=[],
                relationships=[],
                score=0.0,
                metadata={"query": query, "method": "graph_traversal"}
            )
            
        # Traverse graph to find connected entities
        all_entity_ids = set()
        for seed in seed_entities:
            connected = self.store.get_connected_entities(seed.id, max_depth=traversal_depth)
            all_entity_ids.update(connected)
            
        # Get entities and relationships
        entities = [self.store.get_entity(eid) for eid in all_entity_ids if self.store.get_entity(eid)]
        
        # Get relationships between these entities
        relationships = []
        for entity in entities:
            rels = self.store.get_entity_relationships(entity.id)
            # Only include relationships where both entities are in our result set
            for rel in rels:
                if rel.source_entity_id in all_entity_ids and rel.target_entity_id in all_entity_ids:
                    if rel not in relationships:
                        relationships.append(rel)
                        
        # Get documents containing these entities
        doc_ids = set()
        for entity in entities:
            doc_ids.update(entity.source_doc_ids)
            
        documents = [self.store.get_document(did) for did in doc_ids if self.store.get_document(did)]
        
        # Rank documents by relevance (number of relevant entities)
        doc_scores = defaultdict(int)
        for entity in entities:
            for doc_id in entity.source_doc_ids:
                doc_scores[doc_id] += 1
                
        # Sort documents by score
        sorted_docs = sorted(documents, key=lambda d: doc_scores.get(d.id, 0), reverse=True)
        
        return GraphQueryResult(
            documents=sorted_docs[:max_results],
            entities=entities,
            relationships=relationships,
            score=len(seed_entities) / max(len(query.split()), 1),  # Simple relevance score
            metadata={
                "query": query,
                "seed_entities": len(seed_entities),
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "method": "graph_traversal"
            }
        )
        
    def _find_query_entities(self, query: str) -> List[Entity]:
        """
        Find entities matching the query text.
        
        Uses fuzzy matching on entity names.
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        matching_entities = []
        
        # Try to match whole query
        matches = self.store.find_entities_by_name(query, fuzzy=True)
        matching_entities.extend(matches)
        
        # Try to match individual words
        for word in words:
            if len(word) > 3:  # Skip short words
                matches = self.store.find_entities_by_name(word, fuzzy=True)
                matching_entities.extend(matches)
                
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in matching_entities:
            if entity.id not in seen:
                seen.add(entity.id)
                unique_entities.append(entity)
                
        return unique_entities
        
    def get_stats(self) -> GraphStats:
        """Get graph statistics."""
        return self.store.get_stats()
