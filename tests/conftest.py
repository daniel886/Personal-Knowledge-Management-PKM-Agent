"""pytest config — keep imports light, avoid network-bound init."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force defaults so tests don't require a real .env
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/chroma_test")
os.environ.setdefault("OBSIDIAN_VAULT_PATH", "./data/test_vault")
