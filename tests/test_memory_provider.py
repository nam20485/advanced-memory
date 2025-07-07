"""
Tests for memory provider functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from advanced_memory.providers.memory_provider import Mem0MemoryProvider
from advanced_memory.models.memory_models import MemoryType, MemoryImportance


@pytest.fixture
def memory_provider():
    """Create a test memory provider instance."""
    provider = Mem0MemoryProvider()
    return provider


@pytest.fixture
def mock_mem0_client():
    """Mock Mem0 client for testing."""
    mock_client = MagicMock()
    return mock_client


class TestMemoryProvider:
    """Test cases for the memory provider."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, memory_provider):
        """Test provider initialization."""
        await memory_provider.initialize()
        assert memory_provider._initialized is True
        assert memory_provider.client is not None
    
    @pytest.mark.asyncio
    async def test_add_interaction_memory(self, memory_provider, mock_mem0_client):
        """Test adding interaction memory."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        mock_mem0_client.add.return_value = {"id": "memory-123", "status": "success"}
        
        conversation_turn = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2024-01-01T00:00:01"}
        ]
        
        result = await memory_provider.add_interaction_memory(
            user_id="test-user",
            conversation_turn=conversation_turn,
            metadata={"session_id": "session-123"}
        )
        
        assert result["status"] == "success"
        assert result["memory_id"] == "memory-123"
        assert result["user_id"] == "test-user"
        mock_mem0_client.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_user_memory(self, memory_provider, mock_mem0_client):
        """Test searching user memory."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        mock_search_results = [
            {
                "id": "memory-1",
                "memory": "User likes Python programming",
                "score": 0.9,
                "relevance": 0.95,
                "created_at": "2024-01-01T00:00:00"
            },
            {
                "id": "memory-2", 
                "memory": "User is learning GraphRAG",
                "score": 0.8,
                "relevance": 0.85,
                "created_at": "2024-01-01T01:00:00"
            }
        ]
        
        mock_mem0_client.search.return_value = mock_search_results
        
        results = await memory_provider.search_user_memory(
            user_id="test-user",
            query="programming",
            limit=5
        )
        
        assert len(results) == 2
        assert results[0].memory.content == "User likes Python programming"
        assert results[0].score == 0.9
        assert results[1].memory.content == "User is learning GraphRAG"
        assert results[1].score == 0.8
        
        mock_mem0_client.search.assert_called_once_with("programming", "test-user", 5)
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, memory_provider, mock_mem0_client):
        """Test getting user profile."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        mock_memories = [
            {"memory": "User prefers Python over Java", "created_at": "2024-01-01T00:00:00"},
            {"memory": "User is a software engineer", "created_at": "2024-01-01T01:00:00"},
            {"memory": "User wants to learn more about AI", "created_at": "2024-01-01T02:00:00"},
            {"memory": "User is expert in web development", "created_at": "2024-01-01T03:00:00"}
        ]
        
        mock_mem0_client.get_all.return_value = mock_memories
        
        profile = await memory_provider.get_user_profile("test-user")
        
        assert profile.user_id == "test-user"
        assert len(profile.preferences) > 0
        assert len(profile.facts) > 0
        assert len(profile.goals) > 0
        assert len(profile.expertise_areas) > 0
        
        mock_mem0_client.get_all.assert_called_once_with("test-user")
    
    @pytest.mark.asyncio
    async def test_get_user_memory(self, memory_provider, mock_mem0_client):
        """Test getting all user memory."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        mock_memories = [
            {
                "id": "memory-1",
                "memory": "First memory",
                "created_at": "2024-01-01T00:00:00"
            },
            {
                "id": "memory-2",
                "memory": "Second memory", 
                "created_at": "2024-01-01T01:00:00"
            }
        ]
        
        mock_mem0_client.get_all.return_value = mock_memories
        
        user_memory = await memory_provider.get_user_memory("test-user")
        
        assert user_memory.user_id == "test-user"
        assert len(user_memory.memories) == 2
        assert user_memory.total_count == 2
        assert user_memory.memories[0].content == "First memory"
        assert user_memory.memories[1].content == "Second memory"
    
    def test_format_conversation_for_storage(self, memory_provider):
        """Test conversation formatting for storage."""
        conversation_turn = [
            {
                "role": "user",
                "content": "Hello there!",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "role": "assistant", 
                "content": "Hi! How can I help you?",
                "timestamp": "2024-01-01T00:00:01"
            }
        ]
        
        formatted = memory_provider._format_conversation_for_storage(conversation_turn)
        
        assert "[2024-01-01T00:00:00] user: Hello there!" in formatted
        assert "[2024-01-01T00:00:01] assistant: Hi! How can I help you?" in formatted
    
    def test_classify_memory_type(self, memory_provider):
        """Test memory type classification."""
        # Test factual memory
        factual_conversation = [
            {"content": "Remember this important fact about the system"}
        ]
        memory_type = memory_provider._classify_memory_type(factual_conversation)
        assert memory_type == MemoryType.FACTUAL
        
        # Test semantic memory
        semantic_conversation = [
            {"content": "I understand the concept of machine learning"}
        ]
        memory_type = memory_provider._classify_memory_type(semantic_conversation)
        assert memory_type == MemoryType.SEMANTIC
        
        # Test working memory
        working_conversation = [
            {"content": "I'm currently working on a Python project"}
        ]
        memory_type = memory_provider._classify_memory_type(working_conversation)
        assert memory_type == MemoryType.WORKING
        
        # Test episodic memory (default)
        episodic_conversation = [
            {"content": "We had a great conversation yesterday"}
        ]
        memory_type = memory_provider._classify_memory_type(episodic_conversation)
        assert memory_type == MemoryType.EPISODIC
    
    def test_assess_memory_importance(self, memory_provider):
        """Test memory importance assessment."""
        # Test high importance
        high_conversation = [
            {"content": "This is very important to remember"}
        ]
        importance = memory_provider._assess_memory_importance(high_conversation)
        assert importance == MemoryImportance.HIGH
        
        # Test medium importance  
        medium_conversation = [
            {"content": "My goal is to learn GraphRAG"}
        ]
        importance = memory_provider._assess_memory_importance(medium_conversation)
        assert importance == MemoryImportance.MEDIUM
        
        # Test low importance
        low_conversation = [
            {"content": "Just a casual comment by the way"}
        ]
        importance = memory_provider._assess_memory_importance(low_conversation)
        assert importance == MemoryImportance.LOW
    
    @pytest.mark.asyncio
    async def test_error_handling(self, memory_provider, mock_mem0_client):
        """Test error handling in memory operations."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        # Simulate an error during memory addition
        mock_mem0_client.add.side_effect = Exception("Connection error")
        
        result = await memory_provider.add_interaction_memory(
            user_id="test-user",
            conversation_turn=[{"role": "user", "content": "test"}]
        )
        
        assert result["status"] == "error"
        assert "Connection error" in result["message"]
    
    @pytest.mark.asyncio
    async def test_search_memory_empty_results(self, memory_provider, mock_mem0_client):
        """Test searching memory with no results."""
        memory_provider.client = mock_mem0_client
        memory_provider._initialized = True
        
        mock_mem0_client.search.return_value = []
        
        results = await memory_provider.search_user_memory(
            user_id="test-user",
            query="nonexistent",
            limit=5
        )
        
        assert len(results) == 0
