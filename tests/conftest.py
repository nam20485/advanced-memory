"""
Test suite for the Advanced Memory System.
"""

import pytest
import asyncio
from typing import Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from advanced_memory.config import Settings
    
    return Settings(
        openai_api_key="test-key",
        neo4j_password="test-password",
        environment="test",
        debug=True
    )
