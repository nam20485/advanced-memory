"""
MCP (Model Context Protocol) related data models.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class MCPToolType(str, Enum):
    """Types of MCP tools available."""
    QUERY_KNOWLEDGE_BASE = "query_knowledge_base"
    ADD_INTERACTION_MEMORY = "add_interaction_memory"
    SEARCH_USER_MEMORY = "search_user_memory"
    GET_USER_PROFILE = "get_user_profile"


class MCPSearchType(str, Enum):
    """Types of GraphRAG search."""
    GLOBAL = "global"
    LOCAL = "local"


class MCPToolCall:
    """Represents an MCP tool call."""
    
    def __init__(
        self,
        tool: MCPToolType,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None
    ):
        self.tool = tool
        self.parameters = parameters
        self.call_id = call_id or f"call_{datetime.utcnow().isoformat()}"


class MCPRequest:
    """Represents an MCP request."""
    
    def __init__(
        self,
        method: str,
        params: Dict[str, Any],
        id: Optional[str] = None
    ):
        self.method = method
        self.params = params
        self.id = id or f"req_{datetime.utcnow().isoformat()}"


class MCPResponse:
    """Represents an MCP response."""
    
    def __init__(
        self,
        result: Any = None,
        error: Optional['MCPError'] = None,
        id: Optional[str] = None
    ):
        self.result = result
        self.error = error
        self.id = id
        self.timestamp = datetime.utcnow()


class MCPError:
    """Represents an MCP error."""
    
    def __init__(
        self,
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.data = data or {}


class MCPTool:
    """Represents an MCP tool definition."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.parameters = parameters


# Tool definitions for the Advanced Memory System
KNOWLEDGE_TOOL = MCPTool(
    name="query_knowledge_base",
    description="Queries the main knowledge base to answer domain-specific questions using GraphRAG",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or query to search for"
            },
            "search_type": {
                "type": "string",
                "enum": ["global", "local"],
                "description": "Type of search - global for broad themes, local for specific entities"
            },
            "user_context": {
                "type": "string",
                "description": "Optional user context to personalize the search"
            }
        },
        "required": ["query", "search_type"]
    }
)

ADD_MEMORY_TOOL = MCPTool(
    name="add_interaction_memory",
    description="Stores a recent conversation turn into the user's long-term memory",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique identifier for the user"
            },
            "conversation_turn": {
                "type": "array",
                "items": {"type": "object"},
                "description": "The conversation turn to store"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata about the interaction"
            }
        },
        "required": ["user_id", "conversation_turn"]
    }
)

SEARCH_MEMORY_TOOL = MCPTool(
    name="search_user_memory",
    description="Performs a semantic search over a specific user's memory to retrieve relevant past interactions",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique identifier for the user"
            },
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["user_id", "query"]
    }
)

USER_PROFILE_TOOL = MCPTool(
    name="get_user_profile",
    description="Retrieves key memories and synthesizes them into a concise user profile",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique identifier for the user"
            }
        },
        "required": ["user_id"]
    }
)

# All available tools
MCP_TOOLS = [
    KNOWLEDGE_TOOL,
    ADD_MEMORY_TOOL,
    SEARCH_MEMORY_TOOL,
    USER_PROFILE_TOOL
]
