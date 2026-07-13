"""Repository: insert helpers + dashboard read queries over :class:`Database`.

Writes go through ``db.execute_write`` (the single writer); reads use the
read-only pool. Rows are returned as plain dicts so callers and API responses
never leak ``aiosqlite.Row`` objects.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .database import Database


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Repository:
    """Data-access methods for products, events, evidence, recipes, alarms."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- products -----------------------------------------------------
    async def insert_product(
        self, product_id: str, trigger_ts: float, state: str, recipe_id: int | None
    ) -> None:
        """Insert a new product row at trigger time."""

        now = _now_iso()

        async def _fn(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO products
                    (product_id, trigger_ts, state, recipe_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO NOTHING
                """,
                (product_id, trigger_ts, state, recipe_id, now, now),
            )

        await self._db.execute_write(_fn)

    async def update_product(
        self,
        product_id: str,
        *,
        state: str | None = None,
        outcome: str | None = None,
        anomaly_score: float | None = None,
        decision_reason: str | None = None,
        model_version: str | None = None,
        timings: dict[str, float] | None = None,
    ) -> None:
        """Patch mutable product columns; ``None`` args are left unchanged."""

        sets: list[str] = ["updated_at = ?"]
        params: list[object] = [_now_iso()]
        if state is not None:
            sets.append("state = ?")
            params.append(state)
        if outcome is not None:
            sets.append("outcome = ?")
            params.append(outcome)
        if anomaly_score is not None:
            sets.append("anomaly_score = ?")
            params.append(anomaly_score)
        if decision_reason is not None:
            sets.append("decision_reason = ?")
            params.append(decision_reason)
        if model_version is not None:
            sets.append("model_version = ?")
            params.append(model_version)
        if timings is not None:
            sets.append("timings_json = ?")
            params.append(json.dumps(timings))
        params.append(product_id)

        sql = f"UPDATE products SET {', '.join(sets)} WHERE product_id = ?"

        async def _fn(conn: aiosqlite.Connection) -> None:
            await conn.execute(sql, tuple(params))

        await self._db.execute_write(_fn)

    async def append_product_event(
        self,
        product_id: str,
        event_type: str,
        *,
        from_state: str | None,
        to_state: str | None,
        ts_wall: str | None,
        ts_mono: float | None,
        payload: dict[str, Any] | None,
    ) -> None:
        """Append an immutable state-transition row to ``product_events``."""

        payload_json = json.dumps(payload) if payload is not None else None

        async def _fn(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO product_events
                    (product_id, event_type, from_state, to_state,
                     ts_wall, ts_mono, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    event_type,
                    from_state,
                    to_state,
                    ts_wall,
                    ts_mono,
                    payload_json,
                    _now_iso(),
                ),
            )

        await self._db.execute_write(_fn)

    async def recent_products(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recently triggered products, newest first."""

        rows = await self._db.fetch_all(
            "SELECT * FROM products ORDER BY trigger_ts DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    async def search_products(
        self,
        *,
        outcome: str | None = None,
        state: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Filtered product search for the traceability view."""

        clauses: list[str] = []
        params: list[object] = []
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("created_at <= ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await self._db.fetch_all(
            f"SELECT * FROM products {where} ORDER BY trigger_ts DESC LIMIT ?",
            tuple(params),
        )
        return [dict(r) for r in rows]

    async def product_detail(self, product_id: str) -> dict[str, Any] | None:
        """Full product record: row + ordered events + evidence."""

        product = _row_to_dict(
            await self._db.fetch_one("SELECT * FROM products WHERE product_id = ?", (product_id,))
        )
        if product is None:
            return None
        event_rows = await self._db.fetch_all(
            "SELECT * FROM product_events WHERE product_id = ? ORDER BY id ASC",
            (product_id,),
        )
        evidence_rows = await self._db.fetch_all(
            "SELECT * FROM evidence WHERE product_id = ? ORDER BY id ASC",
            (product_id,),
        )
        product["events"] = [dict(r) for r in event_rows]
        product["evidence"] = [dict(r) for r in evidence_rows]
        return product

    async def reconciliation(self) -> dict[str, int]:
        """Zero-silent-loss counters computed from the persisted product table.

        ``lost = triggered - (terminal + in_flight)`` and is expected to be 0.
        """

        rows = await self._db.fetch_all(
            "SELECT state, outcome, COUNT(*) AS n FROM products GROUP BY state, outcome"
        )
        terminal_states = {"PASS", "REJECT", "FAULT"}
        total = 0
        by_outcome: dict[str, int] = {"PASS": 0, "REJECT": 0, "FAULT": 0}
        in_flight = 0
        for r in rows:
            n = int(r["n"])
            total += n
            state = r["state"]
            if state in terminal_states:
                by_outcome[state] = by_outcome.get(state, 0) + n
            else:
                in_flight += n
        terminal = sum(by_outcome.values())
        return {
            "triggered": total,
            "pass": by_outcome["PASS"],
            "reject": by_outcome["REJECT"],
            "fault": by_outcome["FAULT"],
            "terminal": terminal,
            "in_flight": in_flight,
            "lost": total - terminal - in_flight,
        }

    # ---- evidence -----------------------------------------------------
    async def insert_evidence(
        self, product_id: str, kind: str, path: str, sha256: str, mime: str = "image/jpeg"
    ) -> None:
        """Record an evidence image row (image bytes live on the filesystem)."""

        async def _fn(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                """
                INSERT INTO evidence (product_id, kind, path, sha256, mime, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (product_id, kind, path, sha256, mime, _now_iso()),
            )

        await self._db.execute_write(_fn)

    async def evidence_for(self, product_id: str) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM evidence WHERE product_id = ? ORDER BY id ASC", (product_id,)
        )
        return [dict(r) for r in rows]

    # ---- recipes ------------------------------------------------------
    async def create_recipe_version(
        self,
        name: str,
        category: str,
        model_name: str,
        anomaly_threshold: float,
        params: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create a new immutable recipe version (never updates an existing row)."""

        params_json = json.dumps(params or {})

        async def _fn(conn: aiosqlite.Connection) -> int:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM recipes WHERE name = ?", (name,)
            )
            row = await cursor.fetchone()
            next_version = int(row[0]) + 1
            cursor = await conn.execute(
                """
                INSERT INTO recipes
                    (name, version, category, model_name, anomaly_threshold,
                     params_json, notes, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    name,
                    next_version,
                    category,
                    model_name,
                    anomaly_threshold,
                    params_json,
                    notes,
                    _now_iso(),
                ),
            )
            return int(cursor.lastrowid)

        recipe_id = await self._db.execute_write(_fn)
        recipe = await self.get_recipe(recipe_id)
        assert recipe is not None
        return recipe

    async def activate_recipe(self, recipe_id: int) -> dict[str, Any]:
        """Activate a recipe version, deactivating all others (single active)."""

        async def _fn(conn: aiosqlite.Connection) -> bool:
            cursor = await conn.execute("SELECT id FROM recipes WHERE id = ?", (recipe_id,))
            if await cursor.fetchone() is None:
                return False
            await conn.execute("UPDATE recipes SET active = 0 WHERE active = 1")
            await conn.execute("UPDATE recipes SET active = 1 WHERE id = ?", (recipe_id,))
            return True

        ok = await self._db.execute_write(_fn)
        if not ok:
            raise KeyError(f"recipe {recipe_id} not found")
        recipe = await self.get_recipe(recipe_id)
        assert recipe is not None
        return recipe

    async def get_recipe(self, recipe_id: int) -> dict[str, Any] | None:
        return _row_to_dict(
            await self._db.fetch_one("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
        )

    async def get_active_recipe(self) -> dict[str, Any] | None:
        return _row_to_dict(
            await self._db.fetch_one("SELECT * FROM recipes WHERE active = 1 LIMIT 1")
        )

    async def list_recipes(self) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all("SELECT * FROM recipes ORDER BY name ASC, version DESC")
        return [dict(r) for r in rows]

    # ---- alarms -------------------------------------------------------
    async def insert_alarm(
        self,
        code: str,
        severity: str,
        source: str,
        message: str,
        product_id: str | None = None,
    ) -> int:
        """Insert an active alarm row; returns its id."""

        async def _fn(conn: aiosqlite.Connection) -> int:
            cursor = await conn.execute(
                """
                INSERT INTO alarms
                    (code, severity, source, message, product_id, active, raised_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (code, severity, source, message, product_id, _now_iso()),
            )
            return int(cursor.lastrowid)

        return await self._db.execute_write(_fn)

    async def clear_alarm(self, alarm_id: int) -> None:
        async def _fn(conn: aiosqlite.Connection) -> None:
            await conn.execute(
                "UPDATE alarms SET active = 0, cleared_at = ? WHERE id = ?",
                (_now_iso(), alarm_id),
            )

        await self._db.execute_write(_fn)

    async def list_alarms(
        self, *, active_only: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        where = "WHERE active = 1" if active_only else ""
        rows = await self._db.fetch_all(
            f"SELECT * FROM alarms {where} ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]


__all__ = ["Repository"]
