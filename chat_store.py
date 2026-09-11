"""Small JSON-backed conversation store for the Streamlit demo.

This is intentionally local and dependency-free. Replace it with SQLite or a
database once the demo needs multiple users or concurrent writes.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


STORE_PATH = Path(__file__).parent / "data" / "conversations.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_conversations() -> dict[str, dict]:
    if not STORE_PATH.exists():
        return {}
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Keep the UI usable if a manually edited/local JSON file is invalid.
        return {}


def save_conversations(conversations: dict[str, dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STORE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(STORE_PATH)


def new_conversation(title: str = "新对话") -> dict:
    now = _now()
    return {
        "id": str(uuid4()),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def touch(conversation: dict) -> None:
    conversation["updated_at"] = _now()
