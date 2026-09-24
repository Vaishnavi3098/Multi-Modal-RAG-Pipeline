"""Short-term conversation memory + long-term persistence with tenant isolation."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-process sliding window of conversation turns."""

    def __init__(self, max_turns: int = 10, max_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def _key(self, tenant_id: str, conversation_id: str) -> str:
        return f"{tenant_id}:{conversation_id}"

    def add(self, tenant_id: str, conversation_id: str, role: str, content: str) -> None:
        key = self._key(tenant_id, conversation_id)
        if key not in self._sessions:
            self._sessions[key] = []
        self._sessions[key].append({"role": role, "content": content})
        # Trim
        while len(self._sessions[key]) > self.max_turns:
            self._sessions[key].pop(0)

    def get(self, tenant_id: str, conversation_id: str) -> List[Dict[str, str]]:
        return list(self._sessions.get(self._key(tenant_id, conversation_id), []))

    def clear(self, tenant_id: str, conversation_id: str) -> None:
        self._sessions.pop(self._key(tenant_id, conversation_id), None)

    def clear_tenant(self, tenant_id: str) -> None:
        keys = [k for k in self._sessions if k.startswith(f"{tenant_id}:")]
        for k in keys:
            del self._sessions[k]


class LongTermMemory:
    """SQLite-backed persistent memory of queries + retrieved context."""

    def __init__(self, db_path: str | Path = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT,
                    conversation_id TEXT,
                    query TEXT,
                    answer TEXT,
                    chunk_ids TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_tenant ON memory(tenant_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_conv ON memory(conversation_id)"
            )

    def store(
        self,
        tenant_id: str,
        query: str,
        answer: str,
        chunk_ids: List[str],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        mid = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory (id, tenant_id, user_id, conversation_id, query, answer, chunk_ids, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    tenant_id,
                    user_id,
                    conversation_id,
                    query,
                    answer,
                    json.dumps(chunk_ids),
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
        return mid

    def get_recent(
        self,
        tenant_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        with self._connect() as conn:
            if conversation_id:
                rows = conn.execute(
                    """
                    SELECT * FROM memory
                    WHERE tenant_id = ? AND conversation_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (tenant_id, conversation_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memory WHERE tenant_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (tenant_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def purge_older_than(self, days: int = 90, tenant_id: Optional[str] = None) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            if tenant_id:
                cur = conn.execute(
                    "DELETE FROM memory WHERE tenant_id = ? AND created_at < ?",
                    (tenant_id, cutoff),
                )
            else:
                cur = conn.execute("DELETE FROM memory WHERE created_at < ?", (cutoff,))
            return cur.rowcount


class MemoryManager:
    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
    ):
        self.short = short_term or ShortTermMemory()
        self.long = long_term or LongTermMemory()

    def add_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        self.short.add(tenant_id, conversation_id, role, content)

    def get_context(
        self, tenant_id: str, conversation_id: str
    ) -> List[Dict[str, str]]:
        return self.short.get(tenant_id, conversation_id)

    def persist_interaction(
        self,
        tenant_id: str,
        query: str,
        answer: str,
        chunk_ids: List[str],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        return self.long.store(
            tenant_id=tenant_id,
            query=query,
            answer=answer,
            chunk_ids=chunk_ids,
            conversation_id=conversation_id,
            user_id=user_id,
        )

    def reset_conversation(self, tenant_id: str, conversation_id: str) -> None:
        self.short.clear(tenant_id, conversation_id)
