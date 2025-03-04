#!/bin/bash

# Initialize default values
ENV_FILE=".env"
BUILD_FLAG=""
DETACH_FLAG="-d"
CACHE_FLAG=""

# Parse command line arguments
COMMAND=""
COMMAND_ARGS=""

# Function to show usage
show_usage() {
  echo "Usage: ./scripts/run_docker.sh [start|stop|restart|build|logs|cli|verify] [options]"
  echo ""
  echo "Options:"
  echo "  --env-file PATH   Specify a custom .env file location"
  echo "  --build           Force rebuild of containers"
  echo "  --detach          Run containers in detached mode (default)"
  echo "  --no-detach       Run containers in foreground"
  echo "  --no-cache        Build without using cache"
  echo ""
  echo "For cli command:"
  echo "  ./scripts/run_docker.sh cli [record_id]"
  exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
  show_usage
fi

# Get the command (first argument)
COMMAND=$1
shift

# Process remaining arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --env-file)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --env-file requires a value"
        exit 1
      fi
      ENV_FILE="$2"
      shift 2
      ;;
    --env-file=*)
      ENV_FILE="${1#*=}"
      shift
      ;;
    --build)
      BUILD_FLAG="--build"
      shift
      ;;
    --detach)
      DETACH_FLAG="-d"
      shift
      ;;
    --no-detach)
      DETACH_FLAG=""
      shift
      ;;
    --no-cache)
      CACHE_FLAG="--no-cache"
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      show_usage
      ;;
    *)
      # Collect remaining args for command (e.g., record ID for cli)
      COMMAND_ARGS="$COMMAND_ARGS $1"
      shift
      ;;
  esac
done

# Navigate to the project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Check if the specified .env file exists
if [ ! -f "$ROOT_DIR/$ENV_FILE" ]; then
  echo "Error: $ENV_FILE file not found."
  echo "Please create one based on config/.env.example in the config directory."
  echo "You can run './scripts/setup_env.sh' to set up your environment."
  exit 1
fi

# Define actions
case $COMMAND in
  start)
    echo "Starting Docker containers..."
    # Use --env-file to explicitly point to the .env file
    if [ -n "$BUILD_FLAG" ]; then
      echo "Rebuilding containers before starting..."
    fi
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" up $BUILD_FLAG $CACHE_FLAG $DETACH_FLAG

    if [ -n "$DETACH_FLAG" ]; then
      echo " ✔ Docker containers started successfully."
      echo " ✔ You can now access the application at http://localhost:8501/"
    fi
    ;;
  stop)
    echo "Stopping Docker containers..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" down
    ;;
  restart)
    echo "Restarting Docker containers..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" restart
    ;;
  build)
    echo "Building Docker images..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" build $CACHE_FLAG
    ;;
  logs)
    echo "Showing Docker logs..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" logs -f
    ;;
  cli)
    echo "Running Command Line Interface..."
    # Extract record ID from command args, trimming whitespace
    RECORD_ID=$(echo $COMMAND_ARGS | xargs)

    # Check if a record ID was provided and it's a number
    if [[ -n "$RECORD_ID" ]] && [[ "$RECORD_ID" =~ ^[0-9]+$ ]]; then
      echo "Analyzing patient record ID: $RECORD_ID"
      # Pass RECORD_ID as an environment variable to the cli service
      docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" --profile cli run -e RECORD_ID=$RECORD_ID cli
    else
      echo "Using default record ID from docker-compose.yml"
      docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" --profile cli run cli
    fi
    ;;
  verify)
    echo "Running dataset verification..."
    docker compose -f "$ROOT_DIR/docker/docker-compose.yml" --env-file "$ROOT_DIR/$ENV_FILE" --profile verify up $BUILD_FLAG $DETACH_FLAG verify
    ;;
  *)
    echo "Unknown command: $COMMAND"
    show_usage
    ;;
esac
