#!/bin/bash
# Script to set up environment variables for Docker

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created from .env.example"
fi

# Prompt for API key
read -p "Enter your OpenAI API key: " api_key

# Prompt for base URL
read -p "Enter your OpenAI base URL (press Enter for default): " base_url

# Update the .env file with the provided API key
sed -i '' "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" .env

# Update the base URL if provided
if [ -n "$base_url" ]; then
    sed -i '' "s|OPENAI_BASE_URL=.*|OPENAI_BASE_URL=$base_url|" .env
    echo "OpenAI base URL updated in .env file"
else
    echo "Using default OpenAI base URL"
fi

echo "Environment variables updated in .env file"
echo "You can now run 'docker compose up'"
