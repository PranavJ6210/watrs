"""
api/v1/recommendations.py
─────────────────────────
GET /api/v1/recommendations — weather-aware tour recommendations.

Rate-limited to 10 req/min (EXTERNAL_API_LIMIT) because this endpoint
triggers calls to the OpenWeatherMap external API.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query, Request

from core.security import limiter, EXTERNAL_API_LIMIT
from services.recommendation import RecommendationResponse, get_recommendations

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


@router.get(
    "",
    response_model=RecommendationResponse,
    summary="Get weather-aware place recommendations",
    description=(
        "Returns scored place recommendations within a radius of the given "
        "coordinates, factoring in tag affinity, live weather comfort, and "
        "hidden-gem score."
    ),
)
@limiter.limit(EXTERNAL_API_LIMIT)
async def recommendations(
    request: Request,
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude of the search centre"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude of the search centre"),
    radius_km: float = Query(default=10.0, gt=0.0, le=500.0, description="Search radius in km"),
    tags: Optional[str] = Query(
        default=None,
        description="Comma-separated preference tags (e.g. 'heritage,nature,photography')",
    ),
) -> RecommendationResponse:
    """
    **Pipeline**

    1. Geospatial filter (``$geoNear``)
    2. Weighted scoring: ``S = 0.4·Vibe + 0.3·Weather + 0.3·Hidden``
    3. Live weather validation for the top-5 candidates (2 s timeout)
    """
    db = request.app.state.db
    user_tags: List[str] = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    return await get_recommendations(
        db=db,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        user_tags=user_tags,
    )
