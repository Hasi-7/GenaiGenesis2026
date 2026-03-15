from __future__ import annotations

from pydantic import BaseModel, Field


class AuthPayload(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)


class UserView(BaseModel):
    id: int
    email: str
    createdAt: float | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserView


class ClaimPayload(BaseModel):
    deviceKey: str
    nickname: str | None = None


class ClaimResponse(BaseModel):
    status: str


class DeviceView(BaseModel):
    deviceKey: str
    displayName: str
    sourceKind: str
    lastSeenAt: float
    sampleCount: int
    overallScore: float
    averageConfidence: float
    focusedShare: float
    dominantState: str
    lastState: str
    notificationCount: int
    stateBreakdown: dict[str, int]


class UnclaimedDeviceView(BaseModel):
    device_key: str
    display_name: str
    source_kind: str
    last_seen_at: float


class OverviewResponse(BaseModel):
    days: int
    sampleCount: int
    deviceCount: int
    notificationCount: int
    averageConfidence: float
    overallScore: float
    focusedShare: float
    negativeShare: float
    stateBreakdown: dict[str, int]


class TimelinePoint(BaseModel):
    day: str
    score: float
    sampleCount: int
    states: dict[str, int]


class FeedbackEventView(BaseModel):
    recordedAt: float
    triggerKind: str
    severity: str
    shouldNotify: bool
    text: str
    deviceKey: str
    deviceName: str
    sourceKind: str


class DeviceFeedbackView(BaseModel):
    recordedAt: float
    triggerKind: str
    severity: str
    shouldNotify: bool
    text: str


class DeviceDetailResponse(DeviceView):
    timeline: list[TimelinePoint]
    recentFeedback: list[DeviceFeedbackView]
