"""
Configuration management for the Advanced Memory System.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", 
        description="OpenAI embedding model"
    )
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Mem0 Configuration
    mem0_api_key: Optional[str] = Field(default=None, description="Mem0 API key")
    mem0_user_id: str = Field(default="default_user", description="Default Mem0 user ID")
    
    # MCP Server Configuration
    mcp_server_host: str = Field(default="0.0.0.0", description="MCP server host")
    mcp_server_port: int = Field(default=8080, description="MCP server port")
    mcp_server_reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")
    
    # Data Paths
    data_input_path: str = Field(default="./data/input", description="Input data path")
    data_output_path: str = Field(default="./data/output", description="Output data path")
    graph_data_path: str = Field(default="./data/graph", description="Graph data path")
    
    # Vector Index Configuration
    vector_index_name: str = Field(
        default="text_embeddings", 
        description="Name of the vector index"
    )
    vector_dimension: int = Field(default=1536, description="Vector dimension")
    
    # GraphRAG Configuration
    graphrag_chunk_size: int = Field(default=1000, description="Text chunk size")
    graphrag_chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    graphrag_community_level: int = Field(default=2, description="Community detection level")
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


# Global settings instance
settings = Settings()
