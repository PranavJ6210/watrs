"""
services/recommendation.py
──────────────────────────
Core recommendation engine for WATRS v2.0.

Pipeline:
  Phase 1 → Geospatial filter  ($geoNear)
  Phase 2 → Weighted scoring    S(L) = W_v·V + W_w·C + W_h·H
  Phase 3 → Live weather validation (top-5 only, 2 s circuit breaker)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from core.config import get_settings
from models.place import Place

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger("watrs.recommendation")
settings = get_settings()

# ── Scoring weights ─────────────────────────────────────────────────────────
W_VIBE = 0.4
W_WEATHER = 0.3
W_HIDDEN = 0.3

# ── Hard-failure thresholds ─────────────────────────────────────────────────
INACCESSIBLE_ROAD = "boat_only"
MIN_COMFORT_SCORE = 0.4

# ── Weather API ─────────────────────────────────────────────────────────────
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_TIMEOUT = 2.0  # seconds


# ═══════════════════════════════════════════════════════════════════════════
# Response models
# ═══════════════════════════════════════════════════════════════════════════

class ScoredPlace(BaseModel):
    """A place enriched with its recommendation score and distance."""

    place: Place
    score: float = Field(..., ge=0.0, description="Final weighted score")
    dist_meters: float = Field(..., ge=0.0, description="Distance from query point (m)")
    weather_fallback: bool = Field(
        default=False,
        description="True when live weather was unavailable and history was used",
    )


class RecommendationResponse(BaseModel):
    """Envelope returned by the recommendation endpoint."""

    results: List[ScoredPlace]
    total_candidates: int = Field(..., description="Places found in geo radius before scoring")
    warnings: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _vibe_score(user_tags: List[str], place_tags: List[str]) -> float:
    """Jaccard similarity between user preference tags and place tags."""
    if not user_tags or not place_tags:
        return 0.0
    s_user = set(t.lower() for t in user_tags)
    s_place = set(t.lower() for t in place_tags)
    intersection = s_user & s_place
    union = s_user | s_place
    return len(intersection) / len(union) if union else 0.0


def _comfort_from_weather(weather_data: Dict[str, Any]) -> float:
    """
    Derive a 0-1 comfort score from OpenWeatherMap response.

    Formula: ``clamp(1.0 - abs(temp - 22) / 28, 0, 1)``
    Halved if rain or thunderstorm conditions are detected.
    """
    try:
        temp = weather_data["main"]["temp"]  # °C (units=metric)
        score = max(0.0, min(1.0, 1.0 - abs(temp - 22.0) / 28.0))

        # Penalize adverse conditions
        conditions = {w.get("main", "").lower() for w in weather_data.get("weather", [])}
        if conditions & {"rain", "thunderstorm", "drizzle"}:
            score *= 0.5

        return round(score, 4)
    except (KeyError, TypeError):
        return 0.5  # safe default on malformed response


async def _fetch_live_weather(lat: float, lon: float) -> Dict[str, Any] | None:
    """
    Call OpenWeatherMap with a strict 2-second timeout.

    Returns the parsed JSON dict or ``None`` on any failure (circuit breaker).
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        async with httpx.AsyncClient(timeout=WEATHER_TIMEOUT) as client:
            resp = await client.get(OWM_BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, httpx.TimeoutException, Exception) as exc:
        logger.warning("Live weather fetch failed (lat=%s, lon=%s): %s", lat, lon, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

async def get_recommendations(
    db: "AsyncIOMotorDatabase",
    lat: float,
    lon: float,
    radius_km: float,
    user_tags: List[str],
) -> RecommendationResponse:
    """
    Full recommendation pipeline:

    1. ``$geoNear`` to find candidates within *radius_km*.
    2. Score each candidate with the weighted formula.
    3. Fetch live weather for the **top 5** and re-score / validate.
    """
    warnings: List[str] = []

    # ── Phase 1: Geospatial filter ──────────────────────────────────────
    pipeline: List[Dict[str, Any]] = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [lon, lat]},
                "distanceField": "dist_meters",
                "maxDistance": radius_km * 1000,  # km → m
                "spherical": True,
            }
        }
    ]

    collection = db["places_live"]
    candidates_raw: List[Dict[str, Any]] = []
    async for doc in collection.aggregate(pipeline):
        candidates_raw.append(doc)

    total_candidates = len(candidates_raw)
    logger.info("Phase 1: %d candidates within %.1f km", total_candidates, radius_km)

    # ── Phase 2: Scoring engine ─────────────────────────────────────────
    scored: List[Dict[str, Any]] = []

    for doc in candidates_raw:
        place = Place(**doc)
        dist_m: float = doc.get("dist_meters", 0.0)

        # Hard failure: inaccessible road
        if place.safety_metadata.road_access.value == INACCESSIBLE_ROAD:
            continue

        vibe = _vibe_score(user_tags, place.watrs_tags)
        # Initial weather score from history (will be refined for top-5)
        weather = place.metrics.weather_comfort_history
        hidden = place.metrics.hidden_percentile

        # Preliminary comfort check against history
        if weather < MIN_COMFORT_SCORE:
            continue  # hard failure

        score = (W_VIBE * vibe) + (W_WEATHER * weather) + (W_HIDDEN * hidden)

        scored.append({
            "place": place,
            "score": round(score, 4),
            "dist_meters": round(dist_m, 2),
            "weather_fallback": False,
            "_weather_comfort": weather,
        })

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Phase 2: %d candidates after scoring & hard-failure filter", len(scored))

    # ── Phase 3: Live weather validation (top 5 only) ───────────────────
    top_n = min(5, len(scored))
    for i in range(top_n):
        entry = scored[i]
        place: Place = entry["place"]
        p_lon, p_lat = place.location.coordinates

        weather_data = await _fetch_live_weather(p_lat, p_lon)

        if weather_data is not None:
            comfort = _comfort_from_weather(weather_data)
        else:
            # Circuit breaker: fallback to stored history
            comfort = place.metrics.weather_comfort_history
            entry["weather_fallback"] = True
            if "Live weather unavailable" not in warnings:
                warnings.append("Live weather unavailable — using historical comfort data for some results")

        # Re-check hard failure with live comfort
        if comfort < MIN_COMFORT_SCORE:
            entry["score"] = 0.0
            continue

        # Recalculate score with live weather
        vibe = _vibe_score(user_tags, place.watrs_tags)
        hidden = place.metrics.hidden_percentile
        entry["score"] = round(
            (W_VIBE * vibe) + (W_WEATHER * comfort) + (W_HIDDEN * hidden), 4
        )
        entry["_weather_comfort"] = comfort

    # Re-sort after live weather adjustment & drop zeroed entries
    scored = [s for s in scored if s["score"] > 0.0]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── Build response ──────────────────────────────────────────────────
    results = [
        ScoredPlace(
            place=entry["place"],
            score=entry["score"],
            dist_meters=entry["dist_meters"],
            weather_fallback=entry["weather_fallback"],
        )
        for entry in scored
    ]

    logger.info("Phase 3 complete: returning %d results", len(results))

    return RecommendationResponse(
        results=results,
        total_candidates=total_candidates,
        warnings=warnings,
    )
