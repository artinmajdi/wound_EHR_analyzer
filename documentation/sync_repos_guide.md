# Managing Public and Private Repository Synchronization

This document outlines the process for maintaining synchronization between our private repository (with sensitive data) and the public-facing repository (sanitized version).

## Repository Details

- **Private Repository:**
  - URL: https://github.com/artinmajdi/Wound-EHR-Analyzer-private.git
  - Main development branch: `main`
  - Public sync branch: `public_repo_sync`

- **Public Repository:**
  - URL: https://github.com/artinmajdi/wound_EHR_analyzer.git
  - Main branch: `main`

## Initial Setup Process

The following steps were taken to initialize the public repository and create the synchronization workflow:

1. **Create the public repository** on GitHub (empty repository)

2. **Set up the public_repo_sync branch** in the private repository:
   ```bash
   # Create and switch to a new branch called public_repo_sync
   git checkout -b public_repo_sync
   ```

3. **Add the public repository as a remote**:
   ```bash
   # Add the public repo as a remote called "public"
   git remote add public https://github.com/artinmajdi/wound_EHR_analyzer.git
   ```

4. **Update .gitignore for the public version**:
   - Modify .gitignore in the public_repo_sync branch to exclude sensitive data
   - Add entries for dataset directories and environment files

5. **Initial synchronization**:
   ```bash
   # Make sure you're on the public_repo_sync branch
   git checkout public_repo_sync

   # Pull the latest from the public repository (if it exists)
   git pull public main

   # Merge changes from the private main branch
   git merge main

   # Push to the public repository
   git push public public_repo_sync:main
   ```

## Regular Sync Workflow

When you need to update the public repository with new changes from the private repository:

1. **Update your private repo's main branch**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Switch to the public sync branch**:
   ```bash
   git checkout public_repo_sync
   ```

3. **Update from public repo** (in case of external contributions):
   ```bash
   git pull public main
   ```

4. **Merge changes from private main**:
   ```bash
   git merge main
   ```

5. **Resolve any merge conflicts** if they occur:
   - Pay special attention to .gitignore and sensitive files
   - Ensure sensitive data is not being committed

6. **Push to the public repository**:
   ```bash
   git push public public_repo_sync:main
   ```

## Managing Sensitive Data

### .gitignore Configuration

The public version's .gitignore includes additional entries to prevent sensitive data from being shared:

```
# Dataset directories
dataset/
datasets/

# Environment variables
.env
.env.*
!.env.example

# Other sensitive data
wound_analysis/utils/logs/*.docx
wound_analysis/utils/logs/*.log
```

### Best Practices

1. **Always verify changes before pushing** to the public repository:
   ```bash
   git diff --cached
   ```

2. **Use git clean to preview** what would be removed:
   ```bash
   # Preview what would be removed
   git clean -xdn

   # Actually remove the files
   git clean -xdf
   ```

3. **If sensitive data was accidentally committed**:
   - Use BFG Repo-Cleaner or git filter-branch to remove the data
   - Force push to overwrite history
   - Change any exposed credentials immediately

## Troubleshooting

### Handling Diverged Branches

If branches have diverged significantly:

```bash
# While on 'public_repo_sync' branch of private repo
git merge main --no-commit
# Review changes
git status
# If satisfied
git commit -m "Merge main into public_repo_sync with sanitized data"
```

### Resolving .gitignore Conflicts

When .gitignore conflicts occur:

1. Accept both sets of ignore patterns
2. Remove duplicates
3. Ensure all sensitive data patterns are included
4. Commit the resolved .gitignore

## Summary of Common Commands

```bash
# Complete sync workflow
git checkout main
git pull origin main
git checkout public_repo_sync
git pull public main
git merge main
# Resolve any conflicts
git push public public_repo_sync:main
```
