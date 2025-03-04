#!/bin/bash

# Check if a command was provided
if [ $# -eq 0 ]; then
  echo "Usage: ./scripts/run_docker.sh [start|stop|restart|build|logs]"
  exit 1
fi

# Navigate to the project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Ensure the .env file exists
if [ ! -f "$ROOT_DIR/.env" ]; then
  echo "Error: .env file not found in project root."
  echo "Please create one based on config/.env.example in the config directory."
  echo "You can run './scripts/setup_env.sh' to set up your environment."
  exit 1
fi

# Define actions
case $1 in
  start)
    echo "Starting Docker containers..."
    # Use --env-file to explicitly point to the .env file
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/.env" up -d
    echo " ✔ Docker containers started successfully."
    echo " ✔ You can now access the application at http://localhost:8501/"
    ;;
  stop)
    echo "Stopping Docker containers..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" down
    ;;
  restart)
    echo "Restarting Docker containers..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/.env" restart
    ;;
  build)
    echo "Building Docker images..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/.env" build
    ;;
  logs)
    echo "Showing Docker logs..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" logs -f
    ;;
  *)
    echo "Unknown command: $1"
    echo "Usage: ./scripts/run_docker.sh [start|stop|restart|build|logs]"
    exit 1
    ;;
esac
