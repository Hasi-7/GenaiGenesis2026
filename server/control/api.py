from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from server.control.models import (
    AuthPayload,
    AuthResponse,
    ClaimPayload,
    ClaimResponse,
    DeviceDetailResponse,
    DeviceView,
    FeedbackEventView,
    OverviewResponse,
    TimelinePoint,
    UnclaimedDeviceView,
    UserView,
)
from server.control.store import UserRecord, get_control_store

app = FastAPI(title="CognitiveSense Control API", version="0.1.0")

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
_AUTH_COOKIE = "cognitivesense_token"
_SNAPSHOT_DIR = Path(get_control_store().db_path).parent / "snapshots"


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth token"
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth header"
        )
    return token


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> UserRecord:
    token = _extract_token(authorization)
    user = get_control_store().get_user_for_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired"
        )
    return user


def get_current_user_from_cookie(
    token: Annotated[str | None, Cookie(alias=_AUTH_COOKIE)] = None,
) -> UserRecord | None:
    if not token:
        return None
    return get_control_store().get_user_for_token(token)


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    days: int = 7,
    device: str | None = None,
    message: str | None = None,
    user: Annotated[UserRecord | None, Depends(get_current_user_from_cookie)] = None,
) -> HTMLResponse:
    selected_days = max(1, min(days, 365))
    if user is None:
        return _TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            {
                "user": None,
                "message": message,
                "days": selected_days,
                "periods": [1, 7, 30, 90],
                "overview": None,
                "devices": [],
                "selected_device": None,
                "timeline": [],
                "feedback": [],
                "unclaimed": get_control_store().list_recent_unclaimed_devices(),
            },
        )

    overview = get_control_store().overview_for_user(
        user_id=user.id, days=selected_days
    )
    devices = get_control_store().list_devices_for_user(
        user_id=user.id, days=selected_days
    )
    selected_key = device or (devices[0].deviceKey if devices else None)
    selected_device = (
        get_control_store().device_detail_for_user(
            user_id=user.id,
            device_key=selected_key,
            days=selected_days,
        )
        if selected_key is not None
        else None
    )
    return _TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "message": message,
            "days": selected_days,
            "periods": [1, 7, 30, 90],
            "overview": overview,
            "devices": devices,
            "selected_device": selected_device,
            "timeline": get_control_store().timeline_for_user(
                user_id=user.id, days=selected_days
            ),
            "feedback": get_control_store().feedback_for_user(
                user_id=user.id, days=selected_days
            )[:8],
            "unclaimed": get_control_store().list_recent_unclaimed_devices(),
        },
    )


@app.post("/register")
def register_form(
    email: Annotated[str, Form()], password: Annotated[str, Form()]
) -> RedirectResponse:
    try:
        get_control_store().register_user(email, password)
        token = get_control_store().authenticate(email, password)
        if token is None:
            raise RuntimeError("failed to create session")
    except Exception as exc:
        return RedirectResponse(
            url=f"/?message={str(exc)}", status_code=status.HTTP_303_SEE_OTHER
        )
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        _AUTH_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30
    )
    return response


@app.post("/login")
def login_form(
    email: Annotated[str, Form()], password: Annotated[str, Form()]
) -> RedirectResponse:
    token = get_control_store().authenticate(email, password)
    if token is None:
        return RedirectResponse(
            url="/?message=Invalid credentials", status_code=status.HTTP_303_SEE_OTHER
        )
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        _AUTH_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30
    )
    return response


@app.post("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(_AUTH_COOKIE)
    return response


@app.post("/claim-device")
def claim_device_form(
    device_key: Annotated[str, Form()],
    nickname: Annotated[str | None, Form()] = None,
    user: Annotated[UserRecord | None, Depends(get_current_user_from_cookie)] = None,
) -> RedirectResponse:
    if user is None:
        return RedirectResponse(
            url="/?message=Sign in first", status_code=status.HTTP_303_SEE_OTHER
        )
    get_control_store().claim_device(
        user_id=user.id, device_key=device_key, nickname=nickname or None
    )
    return RedirectResponse(
        url=f"/?message=Claimed+{device_key}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/devices/{device_key}/rename")
def rename_device_form(
    device_key: str,
    nickname: Annotated[str, Form()],
    user: Annotated[UserRecord | None, Depends(get_current_user_from_cookie)] = None,
) -> RedirectResponse:
    if user is None:
        return RedirectResponse(
            url="/?message=Sign in first",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    try:
        get_control_store().rename_device(
            user_id=user.id,
            device_key=device_key,
            nickname=nickname,
        )
    except Exception as exc:
        return RedirectResponse(
            url=f"/?device={device_key}&message={str(exc)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return RedirectResponse(
        url=f"/?device={device_key}&message=Renamed+{device_key}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/devices/{device_key}/snapshot.jpg")
def device_snapshot(
    device_key: str,
    user: Annotated[UserRecord | None, Depends(get_current_user_from_cookie)] = None,
) -> FileResponse:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in first")
    detail = get_control_store().device_detail_for_user(
        user_id=user.id,
        device_key=device_key,
        days=365,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Device not found")
    path = _SNAPSHOT_DIR / f"{_safe_filename(device_key)}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No snapshot yet")
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: AuthPayload) -> AuthResponse:
    try:
        user = get_control_store().register_user(payload.email, payload.password)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = get_control_store().authenticate(payload.email, payload.password)
    if token is None:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return AuthResponse(token=token, user=UserView(id=user.id, email=user.email))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: AuthPayload) -> AuthResponse:
    token = get_control_store().authenticate(payload.email, payload.password)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    user = get_control_store().get_user_for_token(token)
    if user is None:
        raise HTTPException(status_code=500, detail="Failed to load user")
    return AuthResponse(token=token, user=UserView(id=user.id, email=user.email))


@app.get("/api/me", response_model=UserView)
def me(user: Annotated[UserRecord, Depends(get_current_user)]) -> UserView:
    return UserView(id=user.id, email=user.email, createdAt=user.created_at)


@app.get("/api/devices/unclaimed", response_model=list[UnclaimedDeviceView])
def unclaimed_devices() -> list[UnclaimedDeviceView]:
    return get_control_store().list_recent_unclaimed_devices()


@app.post("/api/devices/claim", response_model=ClaimResponse)
def claim_device(
    payload: ClaimPayload,
    user: Annotated[UserRecord, Depends(get_current_user)],
) -> ClaimResponse:
    get_control_store().claim_device(
        user_id=user.id,
        device_key=payload.deviceKey,
        nickname=payload.nickname,
    )
    return ClaimResponse(status="claimed")


@app.get("/api/devices", response_model=list[DeviceView])
def devices(
    user: Annotated[UserRecord, Depends(get_current_user)],
    days: int = 7,
) -> list[DeviceView]:
    return get_control_store().list_devices_for_user(
        user_id=user.id,
        days=max(1, min(days, 365)),
    )


@app.get("/api/overview", response_model=OverviewResponse)
def overview(
    user: Annotated[UserRecord, Depends(get_current_user)],
    days: int = 7,
) -> OverviewResponse:
    return get_control_store().overview_for_user(
        user_id=user.id,
        days=max(1, min(days, 365)),
    )


@app.get("/api/timeline", response_model=list[TimelinePoint])
def timeline(
    user: Annotated[UserRecord, Depends(get_current_user)],
    days: int = 7,
) -> list[TimelinePoint]:
    return get_control_store().timeline_for_user(
        user_id=user.id,
        days=max(1, min(days, 365)),
    )


@app.get("/api/feedback", response_model=list[FeedbackEventView])
def feedback(
    user: Annotated[UserRecord, Depends(get_current_user)],
    days: int = 7,
) -> list[FeedbackEventView]:
    return get_control_store().feedback_for_user(
        user_id=user.id,
        days=max(1, min(days, 365)),
    )


@app.get("/api/devices/{device_key}", response_model=DeviceDetailResponse)
def device_detail(
    device_key: str,
    user: Annotated[UserRecord, Depends(get_current_user)],
    days: int = 7,
) -> DeviceDetailResponse:
    detail = get_control_store().device_detail_for_user(
        user_id=user.id,
        device_key=device_key,
        days=max(1, min(days, 365)),
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return detail


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "device"
