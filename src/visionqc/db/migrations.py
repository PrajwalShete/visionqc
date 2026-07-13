"""Numbered schema migrations and a simple idempotent runner.

Migrations are ordered ``(version, sql)`` pairs applied in ascending order.
Applied versions are recorded in ``schema_migrations`` so re-running the runner
is a no-op. Migrations must be forward-only and never edited once released.
"""

from __future__ import annotations

import aiosqlite

# Each entry: (version, human name, SQL script). Append-only.
MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "initial_schema",
        """
        CREATE TABLE products (
            product_id      TEXT PRIMARY KEY,
            trigger_ts      REAL NOT NULL,
            state           TEXT NOT NULL,
            outcome         TEXT,
            anomaly_score   REAL,
            decision_reason TEXT,
            recipe_id       INTEGER,
            model_version   TEXT,
            timings_json    TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE TABLE product_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id   TEXT NOT NULL,
            event_type   TEXT NOT NULL,
            from_state   TEXT,
            to_state     TEXT,
            ts_wall      TEXT,
            ts_mono      REAL,
            payload_json TEXT,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX idx_product_events_pid ON product_events(product_id);

        CREATE TABLE evidence (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            kind       TEXT NOT NULL,
            path       TEXT NOT NULL,
            sha256     TEXT NOT NULL,
            mime       TEXT NOT NULL DEFAULT 'image/jpeg',
            created_at TEXT NOT NULL
        );
        CREATE INDEX idx_evidence_pid ON evidence(product_id);

        CREATE TABLE recipes (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT NOT NULL,
            version           INTEGER NOT NULL,
            category          TEXT NOT NULL,
            model_name        TEXT NOT NULL,
            anomaly_threshold REAL NOT NULL,
            params_json       TEXT,
            notes             TEXT,
            active            INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT NOT NULL,
            UNIQUE(name, version)
        );
        CREATE INDEX idx_recipes_active ON recipes(active);

        CREATE TABLE alarms (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            code       TEXT NOT NULL,
            severity   TEXT NOT NULL,
            source     TEXT NOT NULL,
            message    TEXT NOT NULL,
            product_id TEXT,
            active     INTEGER NOT NULL DEFAULT 1,
            raised_at  TEXT NOT NULL,
            cleared_at TEXT
        );
        CREATE INDEX idx_alarms_active ON alarms(active);

        CREATE INDEX idx_products_outcome ON products(outcome);
        CREATE INDEX idx_products_state ON products(state);
        """,
    ),
]


async def run_migrations(conn: aiosqlite.Connection) -> int:
    """Apply any pending migrations on ``conn``. Returns the resulting version.

    Idempotent: already-applied versions are skipped.
    """

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    await conn.commit()

    cursor = await conn.execute("SELECT version FROM schema_migrations")
    applied = {row[0] for row in await cursor.fetchall()}

    current = max(applied) if applied else 0
    for version, name, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
        if version in applied:
            continue
        await conn.executescript(sql)
        await conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) "
            "VALUES (?, ?, datetime('now'))",
            (version, name),
        )
        await conn.commit()
        current = version
    return current


__all__ = ["MIGRATIONS", "run_migrations"]
