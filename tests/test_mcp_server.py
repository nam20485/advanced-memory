"""
Tests for MCP server functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from advanced_memory.mcp_server import MCPServer
from advanced_memory.models.mcp_models import MCPToolType


@pytest.fixture
def mcp_server():
    """Create a test MCP server instance."""
    server = MCPServer()
    # Mock the providers for testing
    server.knowledge_provider = AsyncMock()
    server.memory_provider = AsyncMock()
    return server


@pytest.fixture
def test_client(mcp_server):
    """Create a test client for the MCP server."""
    return TestClient(mcp_server.app)


class TestMCPServer:
    """Test cases for the MCP server."""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Advanced Memory MCP Server"
        assert data["status"] == "running"
    
    def test_health_check(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "providers" in data
    
    def test_list_tools(self, test_client):
        """Test the tools listing endpoint."""
        response = test_client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0
        
        # Check that required tools are present
        tool_names = [tool["name"] for tool in data["tools"]]
        assert "query_knowledge_base" in tool_names
        assert "add_interaction_memory" in tool_names
        assert "search_user_memory" in tool_names
        assert "get_user_profile" in tool_names
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base_tool(self, mcp_server):
        """Test the knowledge base query tool."""
        # Mock the knowledge provider response
        mock_result = MagicMock()
        mock_result.answer = "Test answer"
        mock_result.confidence = 0.9
        mock_result.processing_time_ms = 100
        mock_result.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        
        mcp_server.knowledge_provider.query_knowledge_base.return_value = mock_result
        
        # Execute the tool
        result = await mcp_server._execute_tool(
            "query_knowledge_base",
            {
                "query": "What is GraphRAG?",
                "search_type": "global"
            }
        )
        
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.9
        assert result["processing_time_ms"] == 100
    
    @pytest.mark.asyncio
    async def test_add_interaction_memory_tool(self, mcp_server):
        """Test the add interaction memory tool."""
        # Mock the memory provider response
        mcp_server.memory_provider.add_interaction_memory.return_value = {
            "status": "success",
            "memory_id": "test-id",
            "user_id": "test-user"
        }
        
        # Execute the tool
        result = await mcp_server._execute_tool(
            "add_interaction_memory",
            {
                "user_id": "test-user",
                "conversation_turn": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        )
        
        assert result["status"] == "success"
        assert result["user_id"] == "test-user"
    
    @pytest.mark.asyncio
    async def test_search_user_memory_tool(self, mcp_server):
        """Test the search user memory tool."""
        # Mock the memory provider response
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"memory": "test memory", "score": 0.8}
        
        mcp_server.memory_provider.search_user_memory.return_value = [mock_result]
        
        # Execute the tool
        result = await mcp_server._execute_tool(
            "search_user_memory",
            {
                "user_id": "test-user",
                "query": "hello",
                "limit": 5
            }
        )
        
        assert "memories" in result
        assert result["count"] == 1
        assert result["memories"][0]["memory"] == "test memory"
    
    @pytest.mark.asyncio
    async def test_get_user_profile_tool(self, mcp_server):
        """Test the get user profile tool."""
        # Mock the memory provider response
        mock_profile = MagicMock()
        mock_profile.to_dict.return_value = {
            "user_id": "test-user",
            "preferences": {},
            "facts": ["User likes Python"],
            "goals": ["Learn GraphRAG"]
        }
        
        mcp_server.memory_provider.get_user_profile.return_value = mock_profile
        
        # Execute the tool
        result = await mcp_server._execute_tool(
            "get_user_profile",
            {"user_id": "test-user"}
        )
        
        assert result["user_id"] == "test-user"
        assert "preferences" in result
        assert "facts" in result
        assert "goals" in result
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, mcp_server):
        """Test error handling for unknown tools."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await mcp_server._execute_tool("unknown_tool", {})
    
    def test_mcp_request_handling(self, test_client):
        """Test MCP request handling via POST."""
        request_data = {
            "method": "tools/list",
            "params": {},
            "id": "test-request"
        }
        
        response = test_client.post("/sse", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-request"
        assert "result" in data
        assert "tools" in data["result"]
    
    def test_invalid_mcp_method(self, test_client):
        """Test handling of invalid MCP methods."""
        request_data = {
            "method": "invalid/method",
            "params": {},
            "id": "test-request"
        }
        
        response = test_client.post("/sse", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-request"
        assert "error" in data
        assert data["error"]["code"] == -32601
