"""
MCP Server implementation for the Advanced Memory System.
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .logging_config import logger
from .models.mcp_models import (
    MCPRequest, MCPResponse, MCPError, MCPToolType, MCPSearchType, MCP_TOOLS
)
from .providers.knowledge_provider import GraphRAGKnowledgeProvider
from .providers.memory_provider import Mem0MemoryProvider


class MCPServer:
    """
    MCP Server for the Advanced Memory System.
    
    Provides a FastAPI-based server that implements the Model Context Protocol
    for exposing GraphRAG knowledge and Mem0 memory capabilities to AI agents.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.app = FastAPI(
            title="Advanced Memory MCP Server",
            description="MCP Server combining GraphRAG and Mem0 for advanced agentic memory",
            version="0.1.0",
            docs_url="/docs" if settings.is_development else None,
            redoc_url="/redoc" if settings.is_development else None
        )
        
        # Initialize providers
        self.knowledge_provider = GraphRAGKnowledgeProvider()
        self.memory_provider = Mem0MemoryProvider()
        
        # Setup FastAPI app
        self._setup_middleware()
        self._setup_routes()
        
        # Track active connections
        self.active_connections: List[Any] = []
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                "Request processed",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            return response
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Advanced Memory MCP Server",
                "version": "0.1.0",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "providers": {
                    "knowledge": self.knowledge_provider._initialized,
                    "memory": self.memory_provider._initialized
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools."""
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                    for tool in MCP_TOOLS
                ]
            }
        
        @self.app.post("/mcp/call")
        async def call_tool(request: Dict[str, Any]):
            """Call an MCP tool directly."""
            try:
                tool_name = request.get("tool")
                parameters = request.get("parameters", {})
                
                if not tool_name:
                    raise HTTPException(status_code=400, detail="Tool name is required")
                
                result = await self._execute_tool(tool_name, parameters)
                return {"result": result}
                
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sse")
        async def sse_endpoint(request: Request):
            """Server-Sent Events endpoint for MCP communication."""
            return EventSourceResponse(self._sse_generator(request))
        
        @self.app.post("/sse")
        async def sse_post_endpoint(request: Request):
            """POST endpoint for SSE (alternative)."""
            data = await request.json()
            result = await self._handle_mcp_request(data)
            return JSONResponse(result)
    
    async def _sse_generator(self, request: Request):
        """Generate Server-Sent Events for MCP communication."""
        client_id = f"client_{datetime.utcnow().timestamp()}"
        logger.info(f"New SSE connection: {client_id}")
        
        try:
            # Send initial connection message
            yield {
                "id": "connection",
                "event": "connected",
                "data": json.dumps({
                    "client_id": client_id,
                    "server": "Advanced Memory MCP Server",
                    "tools": [tool.name for tool in MCP_TOOLS]
                })
            }
            
            # Keep connection alive and handle incoming requests
            while True:
                if await request.is_disconnected():
                    break
                
                # In a real implementation, you'd handle incoming messages here
                # For now, we'll just send periodic keepalive messages
                await asyncio.sleep(30)
                yield {
                    "id": "keepalive",
                    "event": "ping",
                    "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                }
                
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
        finally:
            logger.info(f"SSE connection closed: {client_id}")
    
    async def _handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        try:
            # Parse MCP request
            method = request_data.get("method", "")
            params = request_data.get("params", {})
            request_id = request_data.get("id", "")
            
            if method == "tools/call":
                # Extract tool call information
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                
                # Execute the tool
                result = await self._execute_tool(tool_name, arguments)
                
                # Return MCP response
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result) if isinstance(result, dict) else str(result)
                            }
                        ]
                    }
                }
            
            elif method == "tools/list":
                # Return available tools
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.parameters
                            }
                            for tool in MCP_TOOLS
                        ]
                    }
                }
            
            else:
                # Unknown method
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"MCP request handling failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_data.get("id", ""),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute an MCP tool."""
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        
        try:
            if tool_name == "query_knowledge_base":
                query = parameters.get("query", "")
                search_type = MCPSearchType(parameters.get("search_type", "local"))
                user_context = parameters.get("user_context")
                
                result = await self.knowledge_provider.query_knowledge_base(
                    query=query,
                    search_type=search_type,
                    user_context=user_context
                )
                
                return {
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms,
                    "timestamp": result.timestamp.isoformat()
                }
            
            elif tool_name == "add_interaction_memory":
                user_id = parameters.get("user_id", "")
                conversation_turn = parameters.get("conversation_turn", [])
                metadata = parameters.get("metadata")
                
                result = await self.memory_provider.add_interaction_memory(
                    user_id=user_id,
                    conversation_turn=conversation_turn,
                    metadata=metadata
                )
                
                return result
            
            elif tool_name == "search_user_memory":
                user_id = parameters.get("user_id", "")
                query = parameters.get("query", "")
                limit = parameters.get("limit", 5)
                
                results = await self.memory_provider.search_user_memory(
                    user_id=user_id,
                    query=query,
                    limit=limit
                )
                
                return {
                    "memories": [result.to_dict() for result in results],
                    "count": len(results)
                }
            
            elif tool_name == "get_user_profile":
                user_id = parameters.get("user_id", "")
                
                profile = await self.memory_provider.get_user_profile(user_id)
                
                return profile.to_dict()
            
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def startup(self) -> None:
        """Initialize the server and providers."""
        logger.info("Starting Advanced Memory MCP Server...")
        
        try:
            # Initialize providers
            await self.knowledge_provider.initialize()
            await self.memory_provider.initialize()
            
            logger.info("MCP Server startup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP Server: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Shutting down Advanced Memory MCP Server...")
        
        try:
            # Close provider connections
            await self.knowledge_provider.close()
            
            logger.info("MCP Server shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run(self) -> None:
        """Run the MCP server."""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=settings.mcp_server_host,
            port=settings.mcp_server_port,
            reload=settings.mcp_server_reload and settings.is_development,
            log_level=settings.log_level.lower(),
            access_log=True
        )
