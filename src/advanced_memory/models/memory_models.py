"""
Memory-related data models for Mem0 integration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MemoryType(str, Enum):
    """Types of memories stored in the system."""
    WORKING = "working"
    EPISODIC = "episodic"
    FACTUAL = "factual"
    SEMANTIC = "semantic"


class MemoryImportance(str, Enum):
    """Importance levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryMetadata:
    """Metadata associated with a memory entry."""
    
    def __init__(
        self,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        expires_at: Optional[datetime] = None,
        **kwargs: Any
    ):
        self.importance = importance
        self.tags = tags or []
        self.source = source
        self.confidence = confidence
        self.expires_at = expires_at
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = {
            "importance": self.importance.value,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
        }
        
        if self.expires_at:
            result["expires_at"] = self.expires_at.isoformat()
        
        result.update(self.extra)
        return result


class MemoryEntry:
    """Represents a single memory entry."""
    
    def __init__(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: str,
        memory_id: Optional[str] = None,
        metadata: Optional[MemoryMetadata] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.content = content
        self.memory_type = memory_type
        self.user_id = user_id
        self.memory_id = memory_id
        self.metadata = metadata or MemoryMetadata()
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "user_id": self.user_id,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class MemorySearchResult:
    """Represents a search result from memory."""
    
    def __init__(
        self,
        memory: MemoryEntry,
        score: float,
        relevance: float = 1.0,
        snippet: Optional[str] = None
    ):
        self.memory = memory
        self.score = score
        self.relevance = relevance
        self.snippet = snippet or memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "relevance": self.relevance,
            "snippet": self.snippet,
        }


class UserMemory:
    """Represents all memories for a specific user."""
    
    def __init__(
        self,
        user_id: str,
        memories: Optional[List[MemoryEntry]] = None,
        total_count: int = 0,
        created_at: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.memories = memories or []
        self.total_count = total_count
        self.created_at = created_at or datetime.utcnow()
    
    def add_memory(self, memory: MemoryEntry) -> None:
        """Add a memory to the user's collection."""
        self.memories.append(memory)
        self.total_count += 1
    
    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryEntry]:
        """Get memories of a specific type."""
        return [m for m in self.memories if m.memory_type == memory_type]
    
    def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories."""
        sorted_memories = sorted(
            self.memories, 
            key=lambda m: m.updated_at, 
            reverse=True
        )
        return sorted_memories[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user memory to dictionary."""
        return {
            "user_id": self.user_id,
            "memories": [m.to_dict() for m in self.memories],
            "total_count": self.total_count,
            "created_at": self.created_at.isoformat(),
        }


class ConversationTurn:
    """Represents a single turn in a conversation."""
    
    def __init__(
        self,
        role: str,  # "user" or "assistant"
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation turn to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class UserProfile:
    """Represents a synthesized user profile."""
    
    def __init__(
        self,
        user_id: str,
        preferences: Optional[Dict[str, Any]] = None,
        facts: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        conversation_style: Optional[str] = None,
        expertise_areas: Optional[List[str]] = None,
        last_updated: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.preferences = preferences or {}
        self.facts = facts or []
        self.goals = goals or []
        self.conversation_style = conversation_style
        self.expertise_areas = expertise_areas or []
        self.last_updated = last_updated or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "facts": self.facts,
            "goals": self.goals,
            "conversation_style": self.conversation_style,
            "expertise_areas": self.expertise_areas,
            "last_updated": self.last_updated.isoformat(),
        }
