"""
Advanced Memory System - An MCP Server combining GraphRAG and Mem0.

This package provides a sophisticated memory architecture for AI agents,
combining graph-based retrieval augmented generation (GraphRAG) with
persistent memory management (Mem0).
"""

__version__ = "0.1.0"
__author__ = "nam20485"
__email__ = "your.email@example.com"

from .config import Settings
from .mcp_server import MCPServer

__all__ = ["Settings", "MCPServer"]
