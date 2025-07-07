"""
Memory provider using Mem0 for persistent agentic memory.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from mem0 import MemoryClient
except ImportError:
    # Fallback implementation if mem0 is not available
    class MemoryClient:
        def __init__(self, *args, **kwargs):
            pass
        
        def add(self, *args, **kwargs):
            return {"id": "mock", "status": "success"}
        
        def search(self, *args, **kwargs):
            return []
        
        def get_all(self, *args, **kwargs):
            return []

from ..config import settings
from ..models.memory_models import (
    MemoryEntry, MemorySearchResult, UserMemory, UserProfile,
    ConversationTurn, MemoryType, MemoryImportance, MemoryMetadata
)

logger = logging.getLogger(__name__)


class Mem0MemoryProvider:
    """
    Provides Mem0-based memory management for persistent agentic memory.
    
    This provider handles storing, retrieving, and managing memories for
    individual users, supporting different types of memory including
    working, episodic, factual, and semantic memory.
    """
    
    def __init__(self):
        """Initialize the Mem0 memory provider."""
        self.client: Optional[MemoryClient] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Mem0 client."""
        try:
            if settings.mem0_api_key:
                # Use Mem0 cloud service
                self.client = MemoryClient(api_key=settings.mem0_api_key)
            else:
                # Use local/self-hosted Mem0 (fallback)
                logger.warning("No Mem0 API key provided, using mock client")
                self.client = MemoryClient()
            
            self._initialized = True
            logger.info("Mem0 memory provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 provider: {e}")
            # Use mock client as fallback
            self.client = MemoryClient()
            self._initialized = True
    
    async def add_interaction_memory(
        self, 
        user_id: str, 
        conversation_turn: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a conversation turn into the user's memory.
        
        Args:
            user_id: Unique identifier for the user
            conversation_turn: The conversation data to store
            metadata: Optional metadata about the interaction
            
        Returns:
            Dict containing the status of the operation
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert conversation turn to a structured format
            memory_content = self._format_conversation_for_storage(conversation_turn)
            
            # Determine memory type and importance
            memory_type = self._classify_memory_type(conversation_turn)
            importance = self._assess_memory_importance(conversation_turn)
            
            # Create memory metadata
            memory_metadata = MemoryMetadata(
                importance=importance,
                source="conversation",
                tags=["conversation", "interaction"],
                **(metadata or {})
            )
            
            # Store in Mem0
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.add,
                memory_content,
                user_id,
                memory_metadata.to_dict()
            )
            
            logger.info(f"Added memory for user {user_id}: {result}")
            return {"status": "success", "memory_id": result.get("id"), "user_id": user_id}
            
        except Exception as e:
            logger.error(f"Failed to add interaction memory: {e}")
            return {"status": "error", "message": str(e)}
    
    async def search_user_memory(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5
    ) -> List[MemorySearchResult]:
        """
        Search a user's memory for relevant past interactions.
        
        Args:
            user_id: Unique identifier for the user
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of memory search results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Search Mem0
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.search,
                query,
                user_id,
                limit
            )
            
            # Convert to MemorySearchResult objects
            search_results = []
            for result in results:
                memory_entry = MemoryEntry(
                    content=result.get("memory", ""),
                    memory_type=MemoryType.EPISODIC,  # Default type
                    user_id=user_id,
                    memory_id=result.get("id"),
                    created_at=datetime.fromisoformat(result.get("created_at", datetime.utcnow().isoformat()))
                )
                
                search_result = MemorySearchResult(
                    memory=memory_entry,
                    score=result.get("score", 0.0),
                    relevance=result.get("relevance", 1.0)
                )
                search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} memories for user {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search user memory: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Generate a user profile by synthesizing key memories.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserProfile: Synthesized user profile
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all memories for the user
            all_memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_all,
                user_id
            )
            
            # Analyze memories to extract profile information
            preferences = {}
            facts = []
            goals = []
            expertise_areas = []
            conversation_style = None
            
            for memory in all_memories:
                memory_text = memory.get("memory", "").lower()
                
                # Extract preferences
                if "prefer" in memory_text or "like" in memory_text:
                    preferences[f"pref_{len(preferences)}"] = memory.get("memory", "")
                
                # Extract facts
                if any(indicator in memory_text for indicator in ["is", "am", "works at", "lives in"]):
                    facts.append(memory.get("memory", ""))
                
                # Extract goals
                if any(indicator in memory_text for indicator in ["want to", "goal", "planning to"]):
                    goals.append(memory.get("memory", ""))
                
                # Extract expertise
                if any(indicator in memory_text for indicator in ["expert in", "experienced with", "skilled at"]):
                    expertise_areas.append(memory.get("memory", ""))
            
            # Create user profile
            profile = UserProfile(
                user_id=user_id,
                preferences=preferences,
                facts=facts[:10],  # Limit to top 10
                goals=goals[:5],   # Limit to top 5
                conversation_style=conversation_style,
                expertise_areas=expertise_areas[:5]  # Limit to top 5
            )
            
            logger.info(f"Generated profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            # Return empty profile on error
            return UserProfile(user_id=user_id)
    
    async def get_user_memory(self, user_id: str) -> UserMemory:
        """
        Get all memories for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserMemory: All memories for the user
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all memories from Mem0
            all_memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_all,
                user_id
            )
            
            # Convert to MemoryEntry objects
            memory_entries = []
            for memory in all_memories:
                entry = MemoryEntry(
                    content=memory.get("memory", ""),
                    memory_type=MemoryType.EPISODIC,  # Default type
                    user_id=user_id,
                    memory_id=memory.get("id"),
                    created_at=datetime.fromisoformat(memory.get("created_at", datetime.utcnow().isoformat()))
                )
                memory_entries.append(entry)
            
            # Create UserMemory object
            user_memory = UserMemory(
                user_id=user_id,
                memories=memory_entries,
                total_count=len(memory_entries)
            )
            
            logger.info(f"Retrieved {len(memory_entries)} memories for user {user_id}")
            return user_memory
            
        except Exception as e:
            logger.error(f"Failed to get user memory: {e}")
            return UserMemory(user_id=user_id)
    
    def _format_conversation_for_storage(self, conversation_turn: List[Dict[str, Any]]) -> str:
        """Format conversation turn for storage in memory."""
        formatted_parts = []
        
        for turn in conversation_turn:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", datetime.utcnow().isoformat())
            
            formatted_parts.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def _classify_memory_type(self, conversation_turn: List[Dict[str, Any]]) -> MemoryType:
        """Classify the type of memory based on conversation content."""
        # Simple classification logic - can be enhanced with ML
        content = " ".join([turn.get("content", "") for turn in conversation_turn]).lower()
        
        if any(keyword in content for keyword in ["remember", "recall", "fact", "information"]):
            return MemoryType.FACTUAL
        elif any(keyword in content for keyword in ["concept", "understand", "learn", "knowledge"]):
            return MemoryType.SEMANTIC
        elif any(keyword in content for keyword in ["currently", "now", "today", "working on"]):
            return MemoryType.WORKING
        else:
            return MemoryType.EPISODIC
    
    def _assess_memory_importance(self, conversation_turn: List[Dict[str, Any]]) -> MemoryImportance:
        """Assess the importance of a memory based on conversation content."""
        # Simple importance assessment - can be enhanced with ML
        content = " ".join([turn.get("content", "") for turn in conversation_turn]).lower()
        
        if any(keyword in content for keyword in ["important", "critical", "urgent", "remember this"]):
            return MemoryImportance.HIGH
        elif any(keyword in content for keyword in ["goal", "project", "plan", "deadline"]):
            return MemoryImportance.MEDIUM
        elif any(keyword in content for keyword in ["by the way", "just", "casual", "maybe"]):
            return MemoryImportance.LOW
        else:
            return MemoryImportance.MEDIUM
