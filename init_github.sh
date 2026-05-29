#!/usr/bin/env bash
# init_github.sh — initialise local git repo, create a remote on GitHub via gh CLI,
# and push the first commit.
#
# Usage:
#   ./init_github.sh                 # uses defaults (private repo, gh logged-in user)
#   ./init_github.sh -u myuser -n my-pkm -v public
#
# Requires: git, gh (GitHub CLI). Login first via `gh auth login`.

set -euo pipefail

REPO_NAME="Personal-Knowledge-Management-PKM-Agent"
GITHUB_USERNAME="${GITHUB_USERNAME:-}"
VISIBILITY="private"
DEFAULT_BRANCH="main"

while getopts "u:n:v:b:" opt; do
  case "$opt" in
    u) GITHUB_USERNAME="$OPTARG";;
    n) REPO_NAME="$OPTARG";;
    v) VISIBILITY="$OPTARG";;
    b) DEFAULT_BRANCH="$OPTARG";;
    *) echo "Usage: $0 [-u username] [-n repo_name] [-v public|private] [-b branch]" >&2; exit 1;;
  esac
done

command -v git >/dev/null 2>&1 || { echo "❌ git is required" >&2; exit 1; }
command -v gh  >/dev/null 2>&1 || { echo "❌ GitHub CLI (gh) is required: https://cli.github.com/" >&2; exit 1; }

if ! gh auth status >/dev/null 2>&1; then
  echo "🔐 Please run 'gh auth login' first." >&2
  exit 1
fi

if [[ -z "$GITHUB_USERNAME" ]]; then
  GITHUB_USERNAME=$(gh api user --jq .login)
fi

echo "📦 Repo: $GITHUB_USERNAME/$REPO_NAME ($VISIBILITY)"

# 1. local repo
if [[ ! -d .git ]]; then
  git init -b "$DEFAULT_BRANCH"
fi

# Ensure .gitignore exists
[[ -f .gitignore ]] || cat > .gitignore <<'EOF'
__pycache__/
.venv/
data/
logs/
.env
EOF

git add -A

if git diff --cached --quiet; then
  echo "ℹ️  Nothing to commit."
else
  git commit -m "feat: initial commit — PKM Agent (LangGraph + Chroma + Obsidian)"
fi

# 2. remote repo via gh
if gh repo view "$GITHUB_USERNAME/$REPO_NAME" >/dev/null 2>&1; then
  echo "ℹ️  Remote already exists: $GITHUB_USERNAME/$REPO_NAME"
else
  gh repo create "$GITHUB_USERNAME/$REPO_NAME" \
    --"$VISIBILITY" \
    --description "AI-powered personal knowledge management agent with Obsidian sync, vector search and weekly auto reviews" \
    --source=. \
    --remote=origin \
    --push=false
  echo "✅ Created $GITHUB_USERNAME/$REPO_NAME"
fi

# 3. set / update origin
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
else
  git remote add origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
fi

# 4. push
git branch -M "$DEFAULT_BRANCH"
git push -u origin "$DEFAULT_BRANCH"

echo ""
echo "🎉 Done!"
echo "   👉 https://github.com/$GITHUB_USERNAME/$REPO_NAME"
