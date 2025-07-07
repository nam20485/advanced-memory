FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/
COPY README.md ./
COPY .env.example ./

# Create necessary directories
RUN mkdir -p /app/data/input /app/data/output /app/data/graph /app/logs

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uv", "run", "python", "-m", "src.advanced_memory.main"]
