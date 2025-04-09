#!/bin/bash
# Script to set up environment variables from the setup_config directory

# Navigate to the project root directory
cd "$(dirname "$0")/.." || exit
ROOT_DIR=$(pwd)

# Create .env file if it doesn't exist
if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/setup_config/.env.example" "$ROOT_DIR/.env"
    echo ".env file created from setup_config/.env.example"
fi

# Prompt for API key
read -p "Enter your OpenAI API key: " api_key

# Prompt for base URL
read -p "Enter your OpenAI base URL (press Enter for default): " base_url

# Update the .env file with the provided API key
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS requires an empty string for the -i parameter
    sed -i '' "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" "$ROOT_DIR/.env"

    # Update the base URL if provided
    if [ -n "$base_url" ]; then
        sed -i '' "s|OPENAI_BASE_URL=.*|OPENAI_BASE_URL=$base_url|" "$ROOT_DIR/.env"
        echo "OpenAI base URL updated in .env file"
    else
        echo "Using default OpenAI base URL"
    fi
else
    # Linux version
    sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" "$ROOT_DIR/.env"

    # Update the base URL if provided
    if [ -n "$base_url" ]; then
        sed -i "s|OPENAI_BASE_URL=.*|OPENAI_BASE_URL=$base_url|" "$ROOT_DIR/.env"
        echo "OpenAI base URL updated in .env file"
    else
        echo "Using default OpenAI base URL"
    fi
fi

echo "Environment variables updated in .env file"
echo "You can now run './scripts/run_docker.sh start' to start the application"
