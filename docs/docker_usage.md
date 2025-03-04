# Docker Usage Guide

[← Back to Main README](../README.md) | [Documentation Index](index.md)

This guide provides comprehensive instructions for deploying the Wound Management Interpreter LLM application using Docker containers.

## Quick Start

The project includes a dedicated script that handles all Docker operations:

```bash
# Set up environment variables first
./scripts/setup_env.sh

# Start the application
./scripts/run_docker.sh start
```

## Docker Management Script

The `run_docker.sh` script provides a convenient interface for managing Docker containers:

```bash
./scripts/run_docker.sh [command] [options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `start` | Start the application containers |
| `stop` | Stop running containers |
| `restart` | Restart containers |
| `logs` | View container logs |
| `build` | Rebuild containers |
| `status` | Check container status |
| `shell` | Open a shell in the running container |

### Options

| Option | Description |
|--------|-------------|
| `--env-file PATH` | Specify a custom .env file location |
| `--build` | Force rebuild of containers |
| `--detach` | Run containers in detached mode |
| `--no-cache` | Build without using cache |

## Environment Variables

The Docker deployment uses environment variables for configuration. These can be provided in several ways:

### Method 1: Using the Setup Script (Recommended)

```bash
./scripts/setup_env.sh
./scripts/run_docker.sh start
```

### Method 2: Using Environment Variables at Runtime

```bash
OPENAI_API_KEY=your_key ./scripts/run_docker.sh start
```

### Method 3: Using a Custom .env File

```bash
./scripts/run_docker.sh start --env-file /path/to/custom/.env
```

## Volume Mounts

The Docker configuration includes the following volume mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./dataset` | `/app/dataset` | Data storage |
| `./config` | `/app/config` | Configuration files |

### Custom Dataset Location

You can specify a custom dataset location:

```bash
DATASET_PATH=/path/to/custom/dataset ./scripts/run_docker.sh start
```

## Advanced Usage

### Accessing Container Shell

For debugging or advanced operations:

```bash
./scripts/run_docker.sh shell
```

### Viewing Application Logs

```bash
./scripts/run_docker.sh logs
```

### Custom Docker Compose Configuration

For advanced users who need to modify the Docker Compose configuration:

1. Create a custom docker-compose override file
2. Use the `--file` option:

```bash
./scripts/run_docker.sh start --file /path/to/docker-compose.custom.yml
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Permission errors | Ensure proper permissions on mounted volumes |
| Missing environment variables | Verify .env file exists and contains required variables |
| Port conflicts | Check if port 8501 is already in use |
| Container fails to start | Check logs with `./scripts/run_docker.sh logs` |

### Diagnostic Commands

```bash
# Check container status
./scripts/run_docker.sh status

# View detailed logs
./scripts/run_docker.sh logs

# Rebuild containers from scratch
./scripts/run_docker.sh build --no-cache
```

## Performance Optimization

For production deployments, consider the following optimizations:

1. Increase container resources in docker-compose.yml
2. Use a production-grade web server in front of Streamlit
3. Configure appropriate cache settings

## Security Considerations

- Never commit .env files containing API keys
- Use the `--env-file` option to keep sensitive data outside the project directory
- Consider using Docker secrets for production deployments
