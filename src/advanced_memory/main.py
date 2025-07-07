"""
Main entry point for the Advanced Memory System.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

from .logging_config import logger
from .mcp_server import MCPServer
from .config import settings


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for the FastAPI app."""
    # Startup
    server = MCPServer()
    await server.startup()
    yield
    # Shutdown
    await server.shutdown()


async def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting Advanced Memory System...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Create and configure MCP server
        server = MCPServer()
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(server.shutdown())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize and run server
        await server.startup()
        
        logger.info(f"MCP Server running on {settings.mcp_server_host}:{settings.mcp_server_port}")
        
        # Run the server
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
