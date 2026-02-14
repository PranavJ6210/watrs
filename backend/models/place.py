"""
models/place.py
───────────────
Pydantic v2 models for the ``places_live`` MongoDB collection.

Covers:
  • PyObjectId      — BSON ObjectId ↔ str bridge
  • GeoJSONPoint    — geospatial location with lon/lat validation
  • SafetyMetadata  — road-access enum + safety rating
  • PlaceMetrics    — scoring & visit stats
  • Place           — top-level document model
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Tuple

from bson import ObjectId
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    PlainSerializer,
    BeforeValidator,
    ConfigDict,
)


# ═══════════════════════════════════════════════════════════════════════════
# BSON ObjectId ↔ str
# ═══════════════════════════════════════════════════════════════════════════

def _validate_object_id(v: Any) -> str:
    """Accept an ``ObjectId`` or a 24-hex-char string and return ``str``."""
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str) and ObjectId.is_valid(v):
        return v
    raise ValueError(f"Invalid ObjectId: {v!r}")


PyObjectId = Annotated[
    str,
    BeforeValidator(_validate_object_id),
    PlainSerializer(lambda v: str(v), return_type=str),
]
"""Custom type that transparently converts MongoDB ``ObjectId`` ↔ ``str``."""


# ═══════════════════════════════════════════════════════════════════════════
# GeoJSON Point
# ═══════════════════════════════════════════════════════════════════════════

class GeoJSONPoint(BaseModel):
    """
    GeoJSON Point — ``{"type": "Point", "coordinates": [lon, lat]}``.

    MongoDB requires this exact shape for ``2dsphere`` indexes.
    """

    type: str = Field(default="Point", frozen=True)
    coordinates: Tuple[float, float] = Field(
        ...,
        description="[longitude, latitude]",
    )

    @field_validator("type")
    @classmethod
    def _type_must_be_point(cls, v: str) -> str:
        if v != "Point":
            raise ValueError('GeoJSON type must be "Point"')
        return v

    @field_validator("coordinates")
    @classmethod
    def _validate_coordinates(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lon, lat = v
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        return v


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class RoadAccessLevel(str, Enum):
    """How a place can be physically reached."""

    PAVED = "paved"
    UNPAVED = "unpaved"
    FOURWD_ONLY = "4wd_only"
    FOOT_ONLY = "foot_only"
    BOAT_ONLY = "boat_only"
    OFF_ROAD = "off_road"


class SafetyRating(str, Enum):
    """Overall safety assessment for a place."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ═══════════════════════════════════════════════════════════════════════════
# Nested sub-documents
# ═══════════════════════════════════════════════════════════════════════════

class SafetyMetadata(BaseModel):
    """Safety & accessibility information for a place."""

    road_access: RoadAccessLevel = Field(
        ...,
        description="Physical access classification",
    )
    safety_rating: SafetyRating = Field(
        ...,
        description="Overall safety level",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-text safety notes",
    )
    verified_by_human: bool = Field(
        default=True,
        description="False when a SAFETY_ALERT flags this place for review",
    )


class PlaceMetrics(BaseModel):
    """Scoring and engagement metrics."""

    hidden_percentile: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How 'hidden-gem' this place is (0 = popular, 1 = very hidden)",
    )
    visit_count: int = Field(
        default=0,
        ge=0,
        description="Total recorded visits",
    )
    avg_rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Average user rating (0–5)",
    )
    weather_comfort_history: Dict[str, float] = Field(
        default_factory=dict,
        description="Monthly comfort scores {\"Jan\": 0.8, …} — fallback when live weather unavailable",
    )
    community_trust_score: float = Field(
        default=0.0,
        description="Community trust score modified by LIKE (+0.1) / DISLIKE (-0.1) feedback",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Place — top-level document
# ═══════════════════════════════════════════════════════════════════════════

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Place(BaseModel):
    """
    Represents a single document in the ``places_live`` collection.

    The ``id`` field maps to MongoDB's ``_id`` (ObjectId) and is serialized
    as a plain string in API responses.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "name": "Hampi Ruins",
                "description": "UNESCO World Heritage Site in Karnataka",
                "image_url": "https://unsplash.com/photos/example",
                "location": {
                    "type": "Point",
                    "coordinates": [76.4610, 15.3350],
                },
                "watrs_tags": ["heritage", "ruins", "photography"],
                "safety_metadata": {
                    "road_access": "paved",
                    "safety_rating": "high",
                    "notes": "Well-maintained tourist area",
                },
                "metrics": {
                    "hidden_percentile": 0.35,
                    "visit_count": 12400,
                    "avg_rating": 4.7,
                },
            }
        },
    )

    # ── Identity ────────────────────────────────────────────────────────
    id: Optional[PyObjectId] = Field(
        default=None,
        alias="_id",
        description="MongoDB document ID",
    )

    # ── Core fields ─────────────────────────────────────────────────────
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(default=None, max_length=2000)
    image_url: str = Field(..., description="Image URL (Unsplash, Google, etc.)")

    # ── External references (hybrid — all optional) ─────────────────────
    google_place_id: Optional[str] = Field(default=None, description="Google Places ID")
    google_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)

    # ── Geospatial ──────────────────────────────────────────────────────
    location: GeoJSONPoint

    # ── Tags ────────────────────────────────────────────────────────────
    watrs_tags: List[str] = Field(default_factory=list)

    # ── Safety ──────────────────────────────────────────────────────────
    safety_metadata: SafetyMetadata

    # ── Metrics ─────────────────────────────────────────────────────────
    metrics: PlaceMetrics

    # ── Timestamps ──────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # ── Helpers ─────────────────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _set_updated_at(cls, values: Any) -> Any:
        """Auto-bump ``updated_at`` on every parse/creation."""
        if isinstance(values, dict):
            values["updated_at"] = _utcnow()
        return values

    def to_mongo(self) -> dict:
        """Serialize to a dict suitable for ``insert_one`` / ``replace_one``.

        Excludes ``id`` (``_id``) so MongoDB can assign it on insert.
        """
        data = self.model_dump(by_alias=True, exclude_none=True)
        data.pop("_id", None)
        return data
