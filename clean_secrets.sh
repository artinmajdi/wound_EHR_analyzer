#!/bin/bash

# This script will remove sensitive information from Git history

# Make the script exit on error
set -e

echo "Removing sensitive information from Git history..."

# Remove the secrets from the .env and .pypirc files in Git history
git filter-branch --force --index-filter \
  "git ls-files -z '.env' '.pypirc' | xargs -0 \
  git update-index --replace --path-only --stdin < \
  <(git ls-files -s | grep -v '.env\|.pypirc')" \
  --prune-empty --tag-name-filter cat -- --all

echo "Cleaning up temporary files..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done!"
echo "Warning: This has rewritten history. Force push with:"
echo "git push --force --all"
echo "If you're working with collaborators, make sure they're aware of this change."
