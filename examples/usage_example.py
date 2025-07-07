"""
Example usage of the Advanced Memory System MCP Server.
"""

import asyncio
import json
import httpx
from typing import Dict, Any


class AdvancedMemoryClient:
    """
    Example client for interacting with the Advanced Memory MCP Server.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            
        Returns:
            Tool response
        """
        response = await self.client.post(
            f"{self.base_url}/mcp/call",
            json={
                "tool": tool_name,
                "parameters": parameters
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def query_knowledge(
        self, 
        query: str, 
        search_type: str = "local",
        user_context: str = None
    ) -> str:
        """
        Query the knowledge base.
        
        Args:
            query: Question to ask
            search_type: "global" or "local"
            user_context: Optional user context
            
        Returns:
            Answer from the knowledge base
        """
        params = {
            "query": query,
            "search_type": search_type
        }
        if user_context:
            params["user_context"] = user_context
        
        result = await self.call_tool("query_knowledge_base", params)
        return result["result"]["answer"]
    
    async def add_memory(
        self, 
        user_id: str, 
        conversation: list,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add a conversation to user memory.
        
        Args:
            user_id: User identifier
            conversation: List of conversation turns
            metadata: Optional metadata
            
        Returns:
            Memory addition result
        """
        params = {
            "user_id": user_id,
            "conversation_turn": conversation
        }
        if metadata:
            params["metadata"] = metadata
        
        result = await self.call_tool("add_interaction_memory", params)
        return result["result"]
    
    async def search_memory(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5
    ) -> list:
        """
        Search user memory.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of memory search results
        """
        result = await self.call_tool("search_user_memory", {
            "user_id": user_id,
            "query": query,
            "limit": limit
        })
        return result["result"]["memories"]
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile
        """
        result = await self.call_tool("get_user_profile", {
            "user_id": user_id
        })
        return result["result"]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def example_usage():
    """
    Example usage of the Advanced Memory System.
    """
    client = AdvancedMemoryClient()
    
    try:
        print("=== Advanced Memory System Example ===\n")
        
        # 1. Query Knowledge Base
        print("1. Querying Knowledge Base (Global Search):")
        answer = await client.query_knowledge(
            query="What are the main themes in the dataset?",
            search_type="global"
        )
        print(f"Answer: {answer}\n")
        
        # 2. Query Knowledge Base (Local Search)
        print("2. Querying Knowledge Base (Local Search):")
        answer = await client.query_knowledge(
            query="Tell me about GraphRAG methodology",
            search_type="local",
            user_context="User is learning about AI and graph technologies"
        )
        print(f"Answer: {answer}\n")
        
        # 3. Add Memory
        print("3. Adding User Memory:")
        conversation = [
            {
                "role": "user",
                "content": "I'm interested in learning about graph databases",
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "role": "assistant", 
                "content": "Graph databases are excellent for managing connected data. Neo4j is a popular choice.",
                "timestamp": "2024-01-01T10:00:05Z"
            }
        ]
        
        memory_result = await client.add_memory(
            user_id="user123",
            conversation=conversation,
            metadata={"session_id": "session_001", "topic": "graph_databases"}
        )
        print(f"Memory added: {memory_result}\n")
        
        # 4. Search Memory
        print("4. Searching User Memory:")
        memories = await client.search_memory(
            user_id="user123",
            query="graph databases",
            limit=3
        )
        print("Found memories:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory['memory']['content'][:100]}... (Score: {memory['score']})")
        print()
        
        # 5. Get User Profile
        print("5. Getting User Profile:")
        profile = await client.get_user_profile("user123")
        print("User Profile:")
        print(f"  - Facts: {len(profile.get('facts', []))} items")
        print(f"  - Preferences: {len(profile.get('preferences', {}))} items")
        print(f"  - Goals: {len(profile.get('goals', []))} items")
        print(f"  - Expertise Areas: {len(profile.get('expertise_areas', []))} items")
        print()
        
        # 6. Complex Workflow Example
        print("6. Complex Workflow Example:")
        
        # Add more conversation context
        conversation2 = [
            {
                "role": "user",
                "content": "I want to build a recommendation system using graph technology",
                "timestamp": "2024-01-01T11:00:00Z"
            },
            {
                "role": "assistant",
                "content": "For recommendation systems, you could use GraphRAG to understand user preferences and content relationships, then use graph algorithms to find similar items.",
                "timestamp": "2024-01-01T11:00:10Z"
            }
        ]
        
        await client.add_memory(
            user_id="user123",
            conversation=conversation2,
            metadata={"session_id": "session_002", "topic": "recommendation_systems"}
        )
        
        # Query knowledge with user context from memory
        user_profile = await client.get_user_profile("user123")
        user_context = f"User is interested in {', '.join(user_profile.get('expertise_areas', []))}"
        
        answer = await client.query_knowledge(
            query="How can GraphRAG be used in recommendation systems?",
            search_type="local",
            user_context=user_context
        )
        print(f"Personalized answer: {answer}\n")
        
        print("=== Example Complete ===")
        
    except Exception as e:
        print(f"Error during example: {e}")
    
    finally:
        await client.close()


async def load_test_example():
    """
    Example load testing the MCP server.
    """
    print("=== Load Testing Example ===\n")
    
    clients = [AdvancedMemoryClient() for _ in range(5)]
    
    async def worker(client_id: int, client: AdvancedMemoryClient):
        """Worker function for load testing."""
        try:
            for i in range(3):
                # Query knowledge
                await client.query_knowledge(
                    f"Test query {i} from client {client_id}",
                    search_type="local"
                )
                
                # Add memory
                await client.add_memory(
                    user_id=f"test_user_{client_id}",
                    conversation=[{
                        "role": "user",
                        "content": f"Test message {i} from client {client_id}",
                        "timestamp": "2024-01-01T12:00:00Z"
                    }]
                )
                
                print(f"Client {client_id} completed iteration {i}")
                
        except Exception as e:
            print(f"Client {client_id} error: {e}")
        finally:
            await client.close()
    
    # Run workers concurrently
    tasks = [
        asyncio.create_task(worker(i, client)) 
        for i, client in enumerate(clients)
    ]
    
    await asyncio.gather(*tasks)
    print("\n=== Load Test Complete ===")


if __name__ == "__main__":
    # Run basic example
    asyncio.run(example_usage())
    
    # Uncomment to run load test
    # asyncio.run(load_test_example())
