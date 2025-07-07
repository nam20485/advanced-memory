#!/usr/bin/env bash

# Setup script for the Advanced Memory System
# This script sets up the development environment and initializes the system

set -e  # Exit on any error

echo "🚀 Setting up Advanced Memory System..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python; then
    echo "❌ Python is required but not installed. Please install Python 3.11+"
    exit 1
fi

if ! command_exists docker; then
    echo "❌ Docker is required but not installed. Please install Docker"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is required but not installed. Please install Docker Compose"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install uv if not present
if ! command_exists uv; then
    echo "📦 Installing uv..."
    pip install uv
fi

# Create directories
echo "📁 Creating data directories..."
mkdir -p data/input data/output data/graph data/neo4j/{data,logs,import} data/prometheus logs

# Setup Python environment
echo "🐍 Setting up Python environment..."
uv sync

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before starting the system"
fi

# Setup Terraform if terraform.tfvars doesn't exist
if [ ! -f infrastructure/terraform.tfvars ]; then
    echo "🏗️  Creating Terraform configuration..."
    cp infrastructure/terraform.tfvars.example infrastructure/terraform.tfvars
    echo "⚠️  Please edit infrastructure/terraform.tfvars with your configuration"
fi

# Install pre-commit hooks for development
if [ "$1" = "--dev" ]; then
    echo "🔧 Setting up development environment..."
    uv run pre-commit install
    echo "✅ Development environment ready"
fi

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Start the system:"
echo "   - With Docker Compose: docker-compose up -d"
echo "   - With Terraform: cd infrastructure && terraform apply"
echo "3. Access the system:"
echo "   - MCP Server: http://localhost:8080"
echo "   - Neo4j Browser: http://localhost:7474"
echo "   - API Documentation: http://localhost:8080/docs"
echo ""
echo "For development:"
echo "- Run tests: uv run pytest"
echo "- Start in dev mode: uv run python -m src.advanced_memory.main"
echo "- View logs: docker-compose logs -f"
