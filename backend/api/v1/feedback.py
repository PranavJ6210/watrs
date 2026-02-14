"""
api/v1/feedback.py
──────────────────
POST /api/v1/feedback/{place_id} — community feedback loop for places.

Supports LIKE, DISLIKE, and SAFETY_ALERT actions with atomic MongoDB
updates. Rate-limited to 10 req/min to prevent vote-manipulation spam.
"""

from __future__ import annotations

from enum import Enum

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.security import limiter, EXTERNAL_API_LIMIT

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class FeedbackType(str, Enum):
    """Allowed feedback actions."""

    LIKE = "like"
    DISLIKE = "dislike"
    SAFETY_ALERT = "safety_alert"


class FeedbackRequest(BaseModel):
    """Request body for the feedback endpoint."""

    feedback_type: FeedbackType = Field(
        ..., description="Type of feedback to submit"
    )


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after processing feedback."""

    place_id: str
    action: FeedbackType
    success: bool = True
    message: str = "Feedback recorded"


# ── Atomic update builders ──────────────────────────────────────────────────

_UPDATE_MAP = {
    FeedbackType.LIKE: {
        "$inc": {"metrics.community_trust_score": 0.1},
    },
    FeedbackType.DISLIKE: {
        "$inc": {"metrics.community_trust_score": -0.1},
    },
    FeedbackType.SAFETY_ALERT: {
        "$set": {"safety_metadata.verified_by_human": False},
    },
}


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.post(
    "/{place_id}",
    response_model=FeedbackResponse,
    summary="Submit feedback for a place",
    description=(
        "Atomically updates a place's community trust score or flags it "
        "for safety review. Rate-limited to 10 requests/minute."
    ),
)
@limiter.limit(EXTERNAL_API_LIMIT)
async def submit_feedback(
    request: Request,
    place_id: str,
    body: FeedbackRequest,
) -> FeedbackResponse:
    """
    **Actions**

    - ``LIKE``  → ``$inc metrics.community_trust_score +0.1``
    - ``DISLIKE`` → ``$inc metrics.community_trust_score -0.1``
    - ``SAFETY_ALERT`` → ``$set safety_metadata.verified_by_human = false``
    """
    # Validate ObjectId format
    if not ObjectId.is_valid(place_id):
        raise HTTPException(status_code=400, detail="Invalid place_id format")

    db = request.app.state.db
    collection = db["places_live"]

    update = _UPDATE_MAP[body.feedback_type]
    result = await collection.update_one(
        {"_id": ObjectId(place_id)},
        update,
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Place not found")

    message_map = {
        FeedbackType.LIKE: "Trust score increased",
        FeedbackType.DISLIKE: "Trust score decreased",
        FeedbackType.SAFETY_ALERT: "Place flagged for safety review",
    }

    return FeedbackResponse(
        place_id=place_id,
        action=body.feedback_type,
        message=message_map[body.feedback_type],
    )
