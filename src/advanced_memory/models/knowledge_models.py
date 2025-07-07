"""
Knowledge-related data models for GraphRAG integration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class SearchType(str, Enum):
    """Types of GraphRAG search operations."""
    GLOBAL = "global"
    LOCAL = "local"


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    CAUSED_BY = "caused_by"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    OTHER = "other"


class Entity:
    """Represents an entity in the knowledge graph."""
    
    def __init__(
        self,
        id: str,
        name: str,
        entity_type: EntityType,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ):
        self.id = id
        self.name = name
        self.entity_type = entity_type
        self.description = description or ""
        self.properties = properties or {}
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
            "embedding": self.embedding,
        }


class Relationship:
    """Represents a relationship between entities."""
    
    def __init__(
        self,
        id: str,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        description: Optional[str] = None,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.description = description or ""
        self.weight = weight
        self.properties = properties or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "description": self.description,
            "weight": self.weight,
            "properties": self.properties,
        }


class Community:
    """Represents a community (cluster) in the knowledge graph."""
    
    def __init__(
        self,
        id: str,
        level: int,
        title: str,
        summary: str,
        entities: List[str],
        relationships: Optional[List[str]] = None,
        size: Optional[int] = None,
        weight: float = 1.0
    ):
        self.id = id
        self.level = level
        self.title = title
        self.summary = summary
        self.entities = entities
        self.relationships = relationships or []
        self.size = size or len(entities)
        self.weight = weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert community to dictionary."""
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "entities": self.entities,
            "relationships": self.relationships,
            "size": self.size,
            "weight": self.weight,
        }


class TextChunk:
    """Represents a chunk of text from the source documents."""
    
    def __init__(
        self,
        id: str,
        content: str,
        source_document: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.content = content
        self.source_document = source_document
        self.start_index = start_index
        self.end_index = end_index or len(content)
        self.embedding = embedding
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert text chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source_document": self.source_document,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


class KnowledgeQuery:
    """Represents a query to the knowledge base."""
    
    def __init__(
        self,
        query: str,
        search_type: SearchType,
        user_context: Optional[str] = None,
        max_results: int = 10,
        community_level: Optional[int] = None,
        entity_types: Optional[List[EntityType]] = None
    ):
        self.query = query
        self.search_type = search_type
        self.user_context = user_context
        self.max_results = max_results
        self.community_level = community_level
        self.entity_types = entity_types or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "user_context": self.user_context,
            "max_results": self.max_results,
            "community_level": self.community_level,
            "entity_types": [et.value for et in self.entity_types],
        }


class KnowledgeResult:
    """Represents the result of a knowledge query."""
    
    def __init__(
        self,
        query: KnowledgeQuery,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relationship]] = None,
        communities: Optional[List[Community]] = None,
        confidence: float = 1.0,
        processing_time_ms: Optional[int] = None
    ):
        self.query = query
        self.answer = answer
        self.sources = sources or []
        self.entities = entities or []
        self.relationships = relationships or []
        self.communities = communities or []
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "query": self.query.to_dict(),
            "answer": self.answer,
            "sources": self.sources,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "communities": [c.to_dict() for c in self.communities],
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }
