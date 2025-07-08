# Advanced Memory System

An advanced agentic memory system that combines graph-based retrieval augmented generation (GraphRAG) with persistent memory management (Mem0), implemented as a Model Context Protocol (MCP) server.

## Overview

The Advanced Memory System provides AI agents with:

- **GraphRAG Knowledge Base**: Structured knowledge retrieval using Neo4j graph database
- **Persistent Memory**: User-specific memory management via Mem0
- **MCP Protocol**: Standardized interface for AI agents
- **Scalable Architecture**: Docker-based deployment with Terraform infrastructure

## Features

### GraphRAG Knowledge Core

- Global search for broad thematic queries
- Local search for specific entity-focused queries  
- Neo4j graph database backend
- Community detection and hierarchical summarization
- Vector embeddings for semantic search

### Mem0 Memory Layer

- Working memory for session context
- Episodic memory for conversation history
- Factual memory for user preferences and facts
- Semantic memory for learned patterns
- User profile synthesis

### MCP Server

- FastAPI-based HTTP server
- Server-Sent Events (SSE) support
- RESTful API endpoints
- Comprehensive error handling
- Health monitoring and metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Optional: Mem0 API key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nam20485/advanced-memory.git
   cd advanced-memory
   ```

2. **Set up environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Install dependencies with uv:**

   ```bash
   pip install uv
   uv sync
   ```

4. **Start services with Docker Compose:**

   ```bash
   docker-compose up -d
   ```

5. **Alternatively, use Terraform:**

   ```bash
   cd infrastructure
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your configuration
   terraform init
   terraform apply
   ```

### Basic Usage

The MCP server will be available at `http://localhost:8080` with the following endpoints:

- **Health Check**: `GET /health`
- **List Tools**: `GET /tools`
- **Tool Calls**: `POST /mcp/call`
- **SSE Endpoint**: `GET /sse`

## API Documentation

### Available Tools

#### 1. Query Knowledge Base

```json
{
  "tool": "query_knowledge_base",
  "parameters": {
    "query": "What is GraphRAG?",
    "search_type": "global",
    "user_context": "User is learning about AI"
  }
}
```

#### 2. Add Interaction Memory

```json
{
  "tool": "add_interaction_memory", 
  "parameters": {
    "user_id": "user123",
    "conversation_turn": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ],
    "metadata": {"session_id": "session123"}
  }
}
```

#### 3. Search User Memory

```json
{
  "tool": "search_user_memory",
  "parameters": {
    "user_id": "user123", 
    "query": "preferences",
    "limit": 5
  }
}
```

#### 4. Get User Profile

```json
{
  "tool": "get_user_profile",
  "parameters": {
    "user_id": "user123"
  }
}
```

### Example Responses

#### Knowledge Query Response

```json
{
  "answer": "GraphRAG is a methodology that transforms unstructured text into structured knowledge graphs...",
  "confidence": 0.95,
  "processing_time_ms": 150,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Memory Search Response

```json
{
  "memories": [
    {
      "memory": {
        "content": "User prefers Python programming",
        "memory_type": "factual",
        "created_at": "2024-01-01T10:00:00Z"
      },
      "score": 0.9,
      "relevance": 0.95
    }
  ],
  "count": 1
}
```

## Architecture

### System Components

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agent      │    │   MCP Server     │    │   Neo4j DB      │
│                 │◄──►│                  │◄──►│   (GraphRAG)    │
│  - LangGraph    │    │  - FastAPI       │    │                 │
│  - CrewAI       │    │  - SSE Support   │    └─────────────────┘
│  - Custom       │    │  - Tool Routing  │    
└─────────────────┘    │                  │    ┌─────────────────┐
                       │                  │◄──►│   Mem0 Service  │
                       └──────────────────┘    │   (Memory)      │
                                               └─────────────────┘
```

### Data Flow

1. **Agent Request**: AI agent sends MCP request to server
2. **Tool Routing**: Server routes request to appropriate provider
3. **Knowledge/Memory**: Provider processes request using Neo4j or Mem0
4. **Response**: Structured response returned to agent
5. **Integration**: Agent incorporates context into final output

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `MEM0_API_KEY` | Mem0 API key | Optional |
| `MCP_SERVER_PORT` | Server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Neo4j Configuration

The system requires Neo4j with APOC and Graph Data Science plugins:

```yaml
environment:
  - NEO4J_PLUGINS=["apoc", "graph-data-science"]
  - NEO4J_dbms_memory_heap_max__size=2G
  - NEO4J_dbms_memory_pagecache_size=1G
```

## Development

### Running Tests

```bash
# Install test dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/advanced_memory --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Sort imports  
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/ tests/
```

### Building Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Build docs
uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

## Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f mcp-server

# Scale services
docker-compose up --scale mcp-server=3
```

### Terraform Deployment

```bash
cd infrastructure

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply configuration
terraform apply

# Destroy resources
terraform destroy
```

### Production Considerations

- **Security**: Use secure passwords and API keys
- **Monitoring**: Enable Prometheus and Grafana
- **Backup**: Regular Neo4j database backups
- **Scaling**: Use load balancers for multiple server instances
- **SSL/TLS**: Configure HTTPS for production endpoints

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check Neo4j is running: `docker ps`
   - Verify credentials in environment variables
   - Check Neo4j logs: `docker logs advanced-memory-neo4j`

2. **MCP Server Not Starting**
   - Check port availability: `netstat -an | findstr 8080`
   - Verify environment variables are set
   - Check server logs: `docker logs advanced-memory-mcp-server`

3. **Memory Operations Failing**
   - Verify Mem0 API key (if using cloud service)
   - Check network connectivity
   - Review memory provider logs

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set in .env file
DEBUG=true
LOG_LEVEL=DEBUG

# Or via environment variable
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/advanced-memory.git
cd advanced-memory

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:

- **Issues**: [GitHub Issues](https://github.com/nam20485/advanced-memory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nam20485/advanced-memory/discussions)
- **Documentation**: [Project Wiki](https://github.com/nam20485/advanced-memory/wiki)

## Roadmap

- [ ] Enhanced vector search capabilities
- [ ] Multi-language support
- [ ] Advanced memory consolidation
- [ ] Real-time knowledge graph updates
- [ ] Integration with more LLM providers
- [ ] Web UI for system management
- [ ] Advanced analytics and insights
- [ ] Cloud provider integrations
