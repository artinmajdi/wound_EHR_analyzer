#!/bin/bash

# This script will remove sensitive information from Git history

# Make the script exit on error
set -e

echo "Removing sensitive information from Git history..."

# Use BFG Repo Cleaner if available (much faster than filter-branch)
if command -v bfg &> /dev/null; then
    echo "Using BFG Repo Cleaner..."
    # Need to make sure we have a clean state
    git gc
    bfg --delete-files .env
    bfg --delete-files .pypirc
else
    echo "Using git filter-branch (slower method)..."

    # Remove .env file from history
    git filter-branch --force --index-filter \
      "git rm --cached --ignore-unmatch .env" \
      --prune-empty --tag-name-filter cat -- --all

    # Remove .pypirc file from history
    git filter-branch --force --index-filter \
      "git rm --cached --ignore-unmatch .pypirc" \
      --prune-empty --tag-name-filter cat -- --all
fi

echo "Cleaning up temporary files..."
rm -rf .git/refs/original/ 2>/dev/null || true
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done!"
echo "Warning: This has rewritten history. Force push with:"
echo "git push --force public public_repo_sync:main"
echo "If you're working with collaborators, make sure they're aware of this change."
