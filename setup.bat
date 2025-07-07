@echo off
REM Setup script for the Advanced Memory System on Windows
REM This script sets up the development environment and initializes the system

echo 🚀 Setting up Advanced Memory System...

REM Check prerequisites
echo 📋 Checking prerequisites...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed. Please install Python 3.11+
    exit /b 1
)

docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is required but not installed. Please install Docker
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is required but not installed. Please install Docker Compose
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Install uv if not present
uv --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing uv...
    pip install uv
)

REM Create directories
echo 📁 Creating data directories...
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "data\graph" mkdir data\graph
if not exist "data\neo4j\data" mkdir data\neo4j\data
if not exist "data\neo4j\logs" mkdir data\neo4j\logs
if not exist "data\neo4j\import" mkdir data\neo4j\import
if not exist "data\prometheus" mkdir data\prometheus
if not exist "logs" mkdir logs

REM Setup Python environment
echo 🐍 Setting up Python environment...
uv sync

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env file...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys before starting the system
)

REM Setup Terraform if terraform.tfvars doesn't exist
if not exist "infrastructure\terraform.tfvars" (
    echo 🏗️  Creating Terraform configuration...
    copy infrastructure\terraform.tfvars.example infrastructure\terraform.tfvars
    echo ⚠️  Please edit infrastructure\terraform.tfvars with your configuration
)

REM Install pre-commit hooks for development
if "%1"=="--dev" (
    echo 🔧 Setting up development environment...
    uv run pre-commit install
    echo ✅ Development environment ready
)

echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Start the system:
echo    - With Docker Compose: docker-compose up -d
echo    - With Terraform: cd infrastructure ^&^& terraform apply
echo 3. Access the system:
echo    - MCP Server: http://localhost:8080
echo    - Neo4j Browser: http://localhost:7474
echo    - API Documentation: http://localhost:8080/docs
echo.
echo For development:
echo - Run tests: uv run pytest
echo - Start in dev mode: uv run python -m src.advanced_memory.main
echo - View logs: docker-compose logs -f
