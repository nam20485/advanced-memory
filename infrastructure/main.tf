terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "development"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "mem0_api_key" {
  description = "Mem0 API key (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "neo4j_password" {
  description = "Neo4j database password"
  type        = string
  default     = "neo4jpassword"
  sensitive   = true
}

# Docker provider configuration
provider "docker" {
  host = "npipe:////.//pipe//docker_engine" # Windows
  # host = "unix:///var/run/docker.sock"    # Linux/macOS
}

# Docker network for the application
resource "docker_network" "advanced_memory_network" {
  name = "advanced-memory-${var.environment}"
}

# Neo4j database service
resource "docker_image" "neo4j" {
  name = "neo4j:5.15-community"
}

resource "docker_container" "neo4j" {
  name  = "advanced-memory-neo4j-${var.environment}"
  image = docker_image.neo4j.image_id

  ports {
    internal = 7474
    external = 7474
  }

  ports {
    internal = 7687
    external = 7687
  }

  env = [
    "NEO4J_AUTH=neo4j/${var.neo4j_password}",
    "NEO4J_PLUGINS=[\"apoc\", \"graph-data-science\"]",
    "NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*",
    "NEO4J_dbms_memory_heap_initial__size=1G",
    "NEO4J_dbms_memory_heap_max__size=2G",
    "NEO4J_dbms_memory_pagecache_size=1G"
  ]

  volumes {
    host_path      = "${path.cwd}/data/neo4j/data"
    container_path = "/data"
  }

  volumes {
    host_path      = "${path.cwd}/data/neo4j/logs"
    container_path = "/logs"
  }

  volumes {
    host_path      = "${path.cwd}/data/neo4j/import"
    container_path = "/var/lib/neo4j/import"
  }

  networks_advanced {
    name = docker_network.advanced_memory_network.name
  }

  healthcheck {
    test         = ["CMD", "cypher-shell", "-u", "neo4j", "-p", var.neo4j_password, "RETURN 1"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "40s"
  }

  restart = "unless-stopped"
}

# Build the MCP server image
resource "docker_image" "mcp_server" {
  name = "advanced-memory-mcp-server:${var.environment}"
  build {
    context    = path.cwd
    dockerfile = "Dockerfile"
    tag        = ["advanced-memory-mcp-server:${var.environment}"]
  }
}

# MCP server service
resource "docker_container" "mcp_server" {
  name  = "advanced-memory-mcp-server-${var.environment}"
  image = docker_image.mcp_server.image_id

  ports {
    internal = 8080
    external = 8080
  }

  env = [
    "OPENAI_API_KEY=${var.openai_api_key}",
    "NEO4J_URI=bolt://neo4j:7687",
    "NEO4J_USERNAME=neo4j",
    "NEO4J_PASSWORD=${var.neo4j_password}",
    "MEM0_API_KEY=${var.mem0_api_key}",
    "MCP_SERVER_HOST=0.0.0.0",
    "MCP_SERVER_PORT=8080",
    "LOG_LEVEL=INFO",
    "ENVIRONMENT=${var.environment}"
  ]

  volumes {
    host_path      = "${path.cwd}/data"
    container_path = "/app/data"
  }

  volumes {
    host_path      = "${path.cwd}/logs"
    container_path = "/app/logs"
  }

  networks_advanced {
    name = docker_network.advanced_memory_network.name
  }

  depends_on = [docker_container.neo4j]

  healthcheck {
    test         = ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 3
    start_period = "40s"
  }

  restart = "unless-stopped"
}

# Prometheus monitoring (optional)
resource "docker_image" "prometheus" {
  count = var.environment == "production" ? 1 : 0
  name  = "prom/prometheus:v2.48.0"
}

resource "docker_container" "prometheus" {
  count = var.environment == "production" ? 1 : 0
  name  = "advanced-memory-prometheus-${var.environment}"
  image = docker_image.prometheus[0].image_id

  ports {
    internal = 9090
    external = 9090
  }

  command = [
    "--config.file=/etc/prometheus/prometheus.yml",
    "--storage.tsdb.path=/prometheus",
    "--web.console.libraries=/etc/prometheus/console_libraries",
    "--web.console.templates=/etc/prometheus/consoles",
    "--storage.tsdb.retention.time=200h",
    "--web.enable-lifecycle"
  ]

  volumes {
    host_path      = "${path.cwd}/monitoring/prometheus.yml"
    container_path = "/etc/prometheus/prometheus.yml"
  }

  volumes {
    host_path      = "${path.cwd}/data/prometheus"
    container_path = "/prometheus"
  }

  networks_advanced {
    name = docker_network.advanced_memory_network.name
  }

  restart = "unless-stopped"
}

# Outputs
output "neo4j_uri" {
  description = "Neo4j connection URI"
  value       = "bolt://localhost:7687"
}

output "mcp_server_url" {
  description = "MCP Server URL"
  value       = "http://localhost:8080"
}

output "neo4j_browser_url" {
  description = "Neo4j Browser URL"
  value       = "http://localhost:7474"
}

output "prometheus_url" {
  description = "Prometheus URL (if enabled)"
  value       = var.environment == "production" ? "http://localhost:9090" : "Not enabled in ${var.environment}"
}
