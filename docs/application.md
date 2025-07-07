# advanced-memory applicaton

## Description

This application implements an advanced agentic memory system that combines graph-based retrieval augmented generation (graphRAG) with a memory management system (mem0). The goal is to create a sophisticated memory architecture that enhances the capabilities of agents in processing and recalling information efficiently.

It is implemented as an MCP Server and tools.

See this document for a comprehensive description, and implemenation options, and plans:
[Combining GraphRAG and Mem0](./Combining%20GraphRAG%20and%20Mem0_.md)

## Development Plan

You will implement this application using option #2 descibed in the document: python and Neo4j graph DB. Use docker and docker compose for the services.

### Software frameworks & Languages

* Python
* Neo4j
* Docker
* Docker Compose
* FastAPI
* uv for python package mgmt and projects
* Terraform to dscribe and provision the infrastructure. (Changing the TF provider should allow one to specify deployment environemtn, i.e local docker/compose vs. cloud providers, kubnetes, etc.)
* Swagger/OpenAPI

### Project Structure

GitHub repo created for the project: <http://github.com/nam20485/advanced-memory>

* Include an automated test suite.
* Extensive documenatotionm for API, usage, eyc
* Swagger endpoints page for API.

## Deliverables

* A working, validated advanced-memory application that implements the described architecture.
