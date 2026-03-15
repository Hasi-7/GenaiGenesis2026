from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from control.models import (
    DeviceDetailResponse,
    DeviceFeedbackView,
    DeviceView,
    FeedbackEventView,
    OverviewResponse,
    TimelinePoint,
    UnclaimedDeviceView,
)


@dataclass(slots=True)
class UserRecord:
    id: int
    email: str
    created_at: float


class ControlStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def register_user(self, email: str, password: str) -> UserRecord:
        normalized = email.strip().lower()
        if not normalized or not password:
            raise ValueError("email and password are required")

        password_hash = _hash_password(password)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)",
                (normalized, password_hash, time.time()),
            )
            conn.commit()
            return UserRecord(
                id=_require_int(cur.lastrowid),
                email=normalized,
                created_at=time.time(),
            )

    def authenticate(self, email: str, password: str) -> str | None:
        normalized = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if row is None or not _verify_password(password, str(row["password_hash"])):
            return None

        token = secrets.token_urlsafe(32)
        expires_at = time.time() + 60 * 60 * 24 * 30
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_tokens(token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, _require_int(row["id"]), time.time(), expires_at),
            )
            conn.commit()
        return token

    def get_user_for_token(self, token: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.id, users.email, users.created_at
                FROM auth_tokens
                JOIN users ON users.id = auth_tokens.user_id
                WHERE auth_tokens.token = ? AND auth_tokens.expires_at > ?
                """,
                (token, time.time()),
            ).fetchone()
        if row is None:
            return None
        return UserRecord(
            id=_require_int(row["id"]),
            email=str(row["email"]),
            created_at=float(row["created_at"]),
        )

    def upsert_device(
        self,
        *,
        device_key: str,
        source_kind: str,
        display_name: str,
        last_ip: str | None,
    ) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM devices WHERE device_key = ?",
                (device_key,),
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO devices(
                        device_key,
                        source_kind,
                        display_name,
                        created_at,
                        last_seen_at,
                        last_ip
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (device_key, source_kind, display_name, now, now, last_ip),
                )
                device_id = _require_int(cur.lastrowid)
            else:
                device_id = _require_int(row["id"])
                conn.execute(
                    """
                    UPDATE devices
                    SET source_kind = ?, display_name = ?, last_seen_at = ?, last_ip = ?
                    WHERE id = ?
                    """,
                    (source_kind, display_name, now, last_ip, device_id),
                )
            conn.commit()
        return device_id

    def open_session(
        self,
        *,
        device_key: str,
        source_kind: str,
        display_name: str,
        transport_session_label: str,
        last_ip: str | None,
    ) -> int:
        device_id = self.upsert_device(
            device_key=device_key,
            source_kind=source_kind,
            display_name=display_name,
            last_ip=last_ip,
        )
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO monitoring_sessions(
                    device_id,
                    transport_session_label,
                    source_kind,
                    started_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (device_id, transport_session_label, source_kind, time.time()),
            )
            conn.commit()
            return _require_int(cur.lastrowid)

    def close_session(self, session_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE monitoring_sessions
                SET ended_at = ?
                WHERE id = ? AND ended_at IS NULL
                """,
                (time.time(), session_id),
            )
            conn.commit()

    def add_state_sample(
        self,
        *,
        session_id: int,
        recorded_at: float,
        state_label: str,
        confidence: float,
        indicators: list[str],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_samples(
                    session_id,
                    recorded_at,
                    state_label,
                    confidence,
                    indicators_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    recorded_at,
                    state_label,
                    confidence,
                    json.dumps(indicators),
                ),
            )
            conn.commit()

    def add_feedback_event(
        self,
        *,
        session_id: int,
        recorded_at: float,
        trigger_kind: str,
        severity: str,
        should_notify: bool,
        text: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events(
                    session_id,
                    recorded_at,
                    trigger_kind,
                    severity,
                    should_notify,
                    text
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    recorded_at,
                    trigger_kind,
                    severity,
                    int(should_notify),
                    text,
                ),
            )
            conn.commit()

    def claim_device(
        self, *, user_id: int, device_key: str, nickname: str | None
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE devices
                SET claimed_user_id = ?,
                    claimed_at = ?,
                    nickname = COALESCE(?, nickname)
                WHERE device_key = ?
                """,
                (user_id, time.time(), nickname, device_key),
            )
            conn.commit()

    def rename_device(self, *, user_id: int, device_key: str, nickname: str) -> None:
        cleaned = nickname.strip()
        if not cleaned:
            raise ValueError("nickname is required")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE devices
                SET nickname = ?
                WHERE claimed_user_id = ? AND device_key = ?
                """,
                (cleaned, user_id, device_key),
            )
            conn.commit()

    def list_devices_for_user(self, *, user_id: int, days: int) -> list[DeviceView]:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            devices = conn.execute(
                """
                SELECT
                    d.id,
                    d.device_key,
                    d.display_name,
                    d.nickname,
                    d.source_kind,
                    d.last_seen_at
                FROM devices d
                WHERE d.claimed_user_id = ?
                ORDER BY d.last_seen_at DESC
                """,
                (user_id,),
            ).fetchall()
            summaries: list[DeviceView] = []
            for row in devices:
                summaries.append(self._device_summary(conn, row, cutoff))
            return summaries

    def list_recent_unclaimed_devices(
        self,
        *,
        days: int = 7,
    ) -> list[UnclaimedDeviceView]:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT device_key, display_name, source_kind, last_seen_at
                FROM devices
                WHERE claimed_user_id IS NULL AND last_seen_at >= ?
                ORDER BY last_seen_at DESC
                """,
                (cutoff,),
            ).fetchall()
        return [UnclaimedDeviceView(**dict(row)) for row in rows]

    def overview_for_user(self, *, user_id: int, days: int) -> OverviewResponse:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            device_count = conn.execute(
                "SELECT COUNT(*) FROM devices WHERE claimed_user_id = ?",
                (user_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT ss.state_label, ss.confidence
                FROM devices d
                JOIN monitoring_sessions ms ON ms.device_id = d.id
                JOIN state_samples ss ON ss.session_id = ms.id
                WHERE d.claimed_user_id = ? AND ss.recorded_at >= ?
                """,
                (user_id, cutoff),
            ).fetchall()
            notification_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM devices d
                JOIN monitoring_sessions ms ON ms.device_id = d.id
                JOIN feedback_events fe ON fe.session_id = ms.id
                WHERE d.claimed_user_id = ?
                  AND fe.recorded_at >= ?
                  AND fe.should_notify = 1
                """,
                (user_id, cutoff),
            ).fetchone()
        state_counts: dict[str, int] = {}
        confidences: list[float] = []
        score_total = 0.0
        for row in rows:
            label = str(row["state_label"])
            confidence = float(row["confidence"])
            state_counts[label] = state_counts.get(label, 0) + 1
            confidences.append(confidence)
            score_total += _label_score(label)
        sample_count = len(rows)
        focused_count = state_counts.get("focused", 0)
        negative_count = sum(
            state_counts.get(label, 0)
            for label in ("fatigued", "stressed", "distracted")
        )
        return OverviewResponse(
            days=days,
            sampleCount=sample_count,
            deviceCount=_coerce_int(device_count[0]) if device_count is not None else 0,
            notificationCount=(
                _coerce_int(notification_count[0])
                if notification_count is not None
                else 0
            ),
            averageConfidence=round(sum(confidences) / sample_count, 3)
            if sample_count
            else 0.0,
            overallScore=round(score_total / sample_count, 1) if sample_count else 0.0,
            focusedShare=round(focused_count / sample_count, 3)
            if sample_count
            else 0.0,
            negativeShare=round(negative_count / sample_count, 3)
            if sample_count
            else 0.0,
            stateBreakdown=state_counts,
        )

    def timeline_for_user(self, *, user_id: int, days: int) -> list[TimelinePoint]:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    DATE(ss.recorded_at, 'unixepoch') AS day,
                    ss.state_label,
                    COUNT(*) AS count
                FROM devices d
                JOIN monitoring_sessions ms ON ms.device_id = d.id
                JOIN state_samples ss ON ss.session_id = ms.id
                WHERE d.claimed_user_id = ? AND ss.recorded_at >= ?
                GROUP BY day, ss.state_label
                ORDER BY day ASC
                """,
                (user_id, cutoff),
            ).fetchall()
        grouped: dict[str, TimelinePoint] = {}
        for row in rows:
            day = str(row["day"])
            grouped.setdefault(
                day,
                TimelinePoint(day=day, score=0.0, sampleCount=0, states={}),
            )
            count = _coerce_int(row["count"])
            label = str(row["state_label"])
            grouped[day].states[label] = count
            grouped[day].score += _label_score(label) * count
        result: list[TimelinePoint] = []
        for item in grouped.values():
            total = sum(item.states.values())
            item.score = round(item.score / total, 1) if total else 0.0
            item.sampleCount = total
            result.append(item)
        return result

    def feedback_for_user(
        self,
        *,
        user_id: int,
        days: int,
    ) -> list[FeedbackEventView]:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    fe.recorded_at,
                    fe.trigger_kind,
                    fe.severity,
                    fe.should_notify,
                    fe.text,
                    d.device_key,
                    COALESCE(d.nickname, d.display_name) AS device_name,
                    d.source_kind
                FROM devices d
                JOIN monitoring_sessions ms ON ms.device_id = d.id
                JOIN feedback_events fe ON fe.session_id = ms.id
                WHERE d.claimed_user_id = ? AND fe.recorded_at >= ?
                ORDER BY fe.recorded_at DESC
                LIMIT 50
                """,
                (user_id, cutoff),
            ).fetchall()
        return [
            FeedbackEventView(
                recordedAt=float(row["recorded_at"]),
                triggerKind=str(row["trigger_kind"]),
                severity=str(row["severity"]),
                shouldNotify=bool(row["should_notify"]),
                text=str(row["text"]),
                deviceKey=str(row["device_key"]),
                deviceName=str(row["device_name"]),
                sourceKind=str(row["source_kind"]),
            )
            for row in rows
        ]

    def device_detail_for_user(
        self,
        *,
        user_id: int,
        device_key: str,
        days: int,
    ) -> DeviceDetailResponse | None:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            device = conn.execute(
                """
                SELECT id, device_key, nickname,
                       COALESCE(nickname, display_name) AS display_name,
                       source_kind, last_seen_at
                FROM devices
                WHERE claimed_user_id = ? AND device_key = ?
                """,
                (user_id, device_key),
            ).fetchone()
            if device is None:
                return None
            summary = self._device_summary(conn, device, cutoff)
            feedback = conn.execute(
                """
                SELECT recorded_at, trigger_kind, severity, should_notify, text
                FROM monitoring_sessions ms
                JOIN feedback_events fe ON fe.session_id = ms.id
                WHERE ms.device_id = ? AND fe.recorded_at >= ?
                ORDER BY fe.recorded_at DESC
                LIMIT 20
                """,
                (_coerce_int(device["id"]), cutoff),
            ).fetchall()
            timeline = conn.execute(
                """
                SELECT DATE(ss.recorded_at, 'unixepoch') AS day,
                       ss.state_label,
                       COUNT(*) AS count
                FROM monitoring_sessions ms
                JOIN state_samples ss ON ss.session_id = ms.id
                WHERE ms.device_id = ? AND ss.recorded_at >= ?
                GROUP BY day, ss.state_label
                ORDER BY day ASC
                """,
                (_coerce_int(device["id"]), cutoff),
            ).fetchall()
        return DeviceDetailResponse(
            **summary.model_dump(),
            timeline=_group_timeline_rows(timeline),
            recentFeedback=[
                DeviceFeedbackView(
                    recordedAt=float(row["recorded_at"]),
                    triggerKind=str(row["trigger_kind"]),
                    severity=str(row["severity"]),
                    shouldNotify=bool(row["should_notify"]),
                    text=str(row["text"]),
                )
                for row in feedback
            ],
        )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_key TEXT NOT NULL UNIQUE,
                    source_kind TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    nickname TEXT,
                    claimed_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    last_ip TEXT
                );

                CREATE TABLE IF NOT EXISTS monitoring_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
                    transport_session_label TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL
                );

                CREATE TABLE IF NOT EXISTS state_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL
                        REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
                    recorded_at REAL NOT NULL,
                    state_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    indicators_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL
                        REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
                    recorded_at REAL NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    should_notify INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _device_summary(
        self,
        conn: sqlite3.Connection,
        device_row: sqlite3.Row,
        cutoff: float,
    ) -> DeviceView:
        device_id = _require_int(device_row["id"])
        state_rows = conn.execute(
            """
            SELECT ss.state_label, ss.confidence, ss.recorded_at
            FROM monitoring_sessions ms
            JOIN state_samples ss ON ss.session_id = ms.id
            WHERE ms.device_id = ? AND ss.recorded_at >= ?
            ORDER BY ss.recorded_at DESC
            """,
            (device_id, cutoff),
        ).fetchall()
        feedback_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM monitoring_sessions ms
            JOIN feedback_events fe ON fe.session_id = ms.id
            WHERE ms.device_id = ? AND fe.recorded_at >= ? AND fe.should_notify = 1
            """,
            (device_id, cutoff),
        ).fetchone()

        state_breakdown: dict[str, int] = {}
        confidence_total = 0.0
        score_total = 0.0
        last_state = "unknown"
        for index, row in enumerate(state_rows):
            label = str(row["state_label"])
            state_breakdown[label] = state_breakdown.get(label, 0) + 1
            confidence_total += float(row["confidence"])
            score_total += _label_score(label)
            if index == 0:
                last_state = label
        sample_count = len(state_rows)
        focused_share = (
            state_breakdown.get("focused", 0) / sample_count if sample_count else 0.0
        )
        nickname = (
            str(device_row["nickname"])
            if "nickname" in device_row.keys() and device_row["nickname"] is not None
            else None
        )
        return DeviceView(
            deviceKey=str(device_row["device_key"]),
            displayName=nickname or str(device_row["display_name"]),
            sourceKind=str(device_row["source_kind"]),
            lastSeenAt=float(device_row["last_seen_at"]),
            sampleCount=sample_count,
            overallScore=round(score_total / sample_count, 1) if sample_count else 0.0,
            averageConfidence=round(confidence_total / sample_count, 3)
            if sample_count
            else 0.0,
            focusedShare=round(focused_share, 3),
            dominantState=_dominant_state(state_breakdown),
            lastState=last_state,
            notificationCount=_coerce_int(feedback_row[0])
            if feedback_row is not None
            else 0,
            stateBreakdown=state_breakdown,
        )


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return f"{base64.b64encode(salt).decode()}:{base64.b64encode(digest).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, digest_b64 = stored.split(":", 1)
    except ValueError:
        return False
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(digest_b64)
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return hmac.compare_digest(expected, actual)


def _label_score(label: str) -> float:
    scores = {
        "focused": 100.0,
        "distracted": 60.0,
        "fatigued": 35.0,
        "stressed": 20.0,
        "unknown": 50.0,
    }
    return scores.get(label.lower(), 50.0)


def _group_timeline_rows(rows: list[sqlite3.Row]) -> list[TimelinePoint]:
    grouped: dict[str, TimelinePoint] = {}
    for row in rows:
        day = str(row["day"])
        grouped.setdefault(
            day,
            TimelinePoint(day=day, score=0.0, sampleCount=0, states={}),
        )
        count = _coerce_int(row["count"])
        label = str(row["state_label"])
        grouped[day].states[label] = count
        grouped[day].score += _label_score(label) * count
    result: list[TimelinePoint] = []
    for item in grouped.values():
        total = sum(item.states.values())
        item.score = round(item.score / total, 1) if total else 0.0
        item.sampleCount = total
        result.append(item)
    return result


def _dominant_state(state_breakdown: dict[str, int]) -> str:
    if not state_breakdown:
        return "unknown"
    return max(state_breakdown, key=lambda label: state_breakdown[label])


_control_store_singleton: ControlStore | None = None


def get_control_store() -> ControlStore:
    global _control_store_singleton
    if _control_store_singleton is None:
        db_path = os.environ.get("COGNITIVESENSE_DB_PATH", "data/control.db")
        _control_store_singleton = ControlStore(db_path)
    return _control_store_singleton


def _require_int(value: object) -> int:
    if not isinstance(value, int):
        raise RuntimeError("expected integer value")
    return value


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise RuntimeError("expected numeric value")
