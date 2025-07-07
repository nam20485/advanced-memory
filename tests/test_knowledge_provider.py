"""
Tests for knowledge provider functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from advanced_memory.providers.knowledge_provider import GraphRAGKnowledgeProvider
from advanced_memory.models.knowledge_models import SearchType, EntityType


@pytest.fixture
def knowledge_provider():
    """Create a test knowledge provider instance."""
    provider = GraphRAGKnowledgeProvider()
    return provider


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    return mock_driver, mock_session


class TestKnowledgeProvider:
    """Test cases for the knowledge provider."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, knowledge_provider):
        """Test provider initialization."""
        with patch('advanced_memory.providers.knowledge_provider.GraphDatabase'):
            with patch('advanced_memory.providers.knowledge_provider.OpenAILLM'):
                with patch('advanced_memory.providers.knowledge_provider.OpenAIEmbeddings'):
                    with patch('advanced_memory.providers.knowledge_provider.VectorCypherRetriever'):
                        with patch('advanced_memory.providers.knowledge_provider.GraphRAG'):
                            await knowledge_provider.initialize()
                            assert knowledge_provider._initialized is True
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base_local(self, knowledge_provider):
        """Test local search functionality."""
        # Mock the GraphRAG system
        mock_graphrag = MagicMock()
        mock_graphrag.search.return_value = MagicMock(answer="Test answer")
        knowledge_provider.graphrag = mock_graphrag
        knowledge_provider._initialized = True
        
        result = await knowledge_provider.query_knowledge_base(
            "What is GraphRAG?",
            SearchType.LOCAL
        )
        
        assert result.answer == "Test answer"
        assert result.query.search_type == SearchType.LOCAL
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base_global(self, knowledge_provider):
        """Test global search functionality."""
        # Mock Neo4j driver and LLM
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Mock community data
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "title": "Test Community",
            "summary": "Test summary",
            "size": 10
        }
        mock_result.__iter__.return_value = [mock_record]
        
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Global search answer"
        
        knowledge_provider.driver = mock_driver
        knowledge_provider.llm = mock_llm
        knowledge_provider._initialized = True
        
        result = await knowledge_provider.query_knowledge_base(
            "What are the main themes?",
            SearchType.GLOBAL
        )
        
        assert result.answer == "Global search answer"
        assert result.query.search_type == SearchType.GLOBAL
    
    @pytest.mark.asyncio
    async def test_index_documents(self, knowledge_provider):
        """Test document indexing functionality."""
        # Mock the indexing pipeline
        mock_pipeline = MagicMock()
        
        with patch('advanced_memory.providers.knowledge_provider.SimpleKGPipeline', return_value=mock_pipeline):
            knowledge_provider.llm = MagicMock()
            knowledge_provider.embedder = MagicMock()
            knowledge_provider.driver = MagicMock()
            knowledge_provider._initialized = True
            
            documents = [
                {"content": "Test document 1", "metadata": {}},
                {"content": "Test document 2", "metadata": {}}
            ]
            
            result = await knowledge_provider.index_documents(documents)
            
            assert result is True
            assert mock_pipeline.run.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_entities(self, knowledge_provider, mock_neo4j_driver):
        """Test entity retrieval functionality."""
        mock_driver, mock_session = mock_neo4j_driver
        
        # Mock entity data
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "e.id": "entity-1",
            "e.name": "Test Entity",
            "e.type": "person",
            "e.description": "A test entity"
        }[key]
        
        mock_session.run.return_value = [mock_record]
        knowledge_provider.driver = mock_driver
        knowledge_provider._initialized = True
        
        entities = await knowledge_provider.get_entities(EntityType.PERSON)
        
        assert len(entities) == 1
        assert entities[0].name == "Test Entity"
        assert entities[0].entity_type == EntityType.PERSON
    
    @pytest.mark.asyncio
    async def test_get_relationships(self, knowledge_provider, mock_neo4j_driver):
        """Test relationship retrieval functionality."""
        mock_driver, mock_session = mock_neo4j_driver
        
        # Mock relationship data
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "source.id": "entity-1",
            "target.id": "entity-2",
            "r.description": "Test relationship",
            "r.weight": 0.8
        }[key]
        
        mock_session.run.return_value = [mock_record]
        knowledge_provider.driver = mock_driver
        knowledge_provider._initialized = True
        
        relationships = await knowledge_provider.get_relationships("entity-1")
        
        assert len(relationships) == 1
        assert relationships[0].source_id == "entity-1"
        assert relationships[0].target_id == "entity-2"
        assert relationships[0].weight == 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling(self, knowledge_provider):
        """Test error handling in query processing."""
        # Simulate an error during query
        knowledge_provider._initialized = True
        knowledge_provider.graphrag = None
        
        result = await knowledge_provider.query_knowledge_base(
            "Test query",
            SearchType.LOCAL
        )
        
        assert "Error processing query" in result.answer
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_close_connection(self, knowledge_provider):
        """Test closing the Neo4j connection."""
        mock_driver = MagicMock()
        knowledge_provider.driver = mock_driver
        
        await knowledge_provider.close()
        
        mock_driver.close.assert_called_once()
