"""
scripts/seed_smart.py
─────────────────────
Seed the ``places_live`` collection with 3 Tamil Nadu archetypes,
fetching historical weather comfort from the Open-Meteo Archive API.

Usage:
    python -m scripts.seed_smart          # from backend/
    python scripts/seed_smart.py          # direct

Climate Engine:
    Fetches 2023 daily max-temp + rain-sum from Open-Meteo, aggregates
    monthly averages, and scores each month as:

        Comfort = 0.5 × T_score + 0.5 × R_score

    Temperature tiers (°C):
        20–30 → 1.0  |  30–33 → 0.85  |  33–36 → 0.65
        36–39 → 0.45 |  >39   → 0.25  |  <20   → 0.7

    Rain tiers (mm/month):
        0–40 → 1.0  |  40–100 → 0.8  |  100–200 → 0.6
        200–350 → 0.4 |  >350  → 0.2
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
from motor.motor_asyncio import AsyncIOMotorClient

# Ensure backend/ is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings
from models.place import (
    GeoJSONPoint,
    Place,
    PlaceMetrics,
    RoadAccessLevel,
    SafetyMetadata,
    SafetyRating,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("watrs.seed")

settings = get_settings()

# ── Month labels ────────────────────────────────────────────────────────────
MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ── Open-Meteo Historical API ──────────────────────────────────────────────
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ═══════════════════════════════════════════════════════════════════════════
# Climate scoring functions
# ═══════════════════════════════════════════════════════════════════════════

def _temp_score(avg_max_temp: float) -> float:
    """Weighted tier score for monthly average max temperature (°C)."""
    if 20.0 <= avg_max_temp <= 30.0:
        return 1.0
    elif 30.0 < avg_max_temp <= 33.0:
        return 0.85
    elif 33.0 < avg_max_temp <= 36.0:
        return 0.65
    elif 36.0 < avg_max_temp <= 39.0:
        return 0.45
    elif avg_max_temp > 39.0:
        return 0.25
    else:  # < 20°C
        return 0.7


def _rain_score(total_rain_mm: float) -> float:
    """Weighted tier score for monthly total rainfall (mm)."""
    if total_rain_mm <= 40.0:
        return 1.0
    elif total_rain_mm <= 100.0:
        return 0.8
    elif total_rain_mm <= 200.0:
        return 0.6
    elif total_rain_mm <= 350.0:
        return 0.4
    else:
        return 0.2


async def calculate_historical_comfort(lat: float, lon: float) -> Dict[str, float]:
    """
    Fetch 2023 daily weather from Open-Meteo and return 12-month comfort map.

    Returns: ``{"Jan": 0.85, "Feb": 0.90, …}``
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": "temperature_2m_max,rain_sum",
        "timezone": "Asia/Kolkata",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    dates: List[str] = data["daily"]["time"]                    # "2023-01-01"
    temps: List[float | None] = data["daily"]["temperature_2m_max"]
    rains: List[float | None] = data["daily"]["rain_sum"]

    # Aggregate by month
    monthly_temps: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
    monthly_rains: Dict[int, List[float]] = {m: [] for m in range(1, 13)}

    for i, date_str in enumerate(dates):
        month = int(date_str.split("-")[1])
        if temps[i] is not None:
            monthly_temps[month].append(temps[i])
        if rains[i] is not None:
            monthly_rains[month].append(rains[i])

    comfort: Dict[str, float] = {}
    for m in range(1, 13):
        avg_temp = sum(monthly_temps[m]) / len(monthly_temps[m]) if monthly_temps[m] else 30.0
        total_rain = sum(monthly_rains[m])

        t_s = _temp_score(avg_temp)
        r_s = _rain_score(total_rain)
        comfort[MONTH_ABBR[m - 1]] = round(0.5 * t_s + 0.5 * r_s, 4)

    return comfort


# ═══════════════════════════════════════════════════════════════════════════
# Seed archetypes
# ═══════════════════════════════════════════════════════════════════════════

ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "Kolli Hills",
        "description": "Lush hill station in Namakkal district with 70 hairpin bends.",
        "image_url": "https://unsplash.com/photos/a-winding-road-through-a-forest-pl3_I8_Y16I",
        "lat": 11.2485,
        "lon": 78.3387,
        "tags": ["Nature", "Trekking"],
        "hidden_percentile": 0.9,
        "road_access": RoadAccessLevel.PAVED,
        "safety_rating": SafetyRating.MODERATE,
    },
    {
        "name": "Marina Beach",
        "description": "India's longest natural urban beach on the Bay of Bengal.",
        "image_url": "https://unsplash.com/photos/an-aerial-view-of-a-beach-and-ocean-cxbT0tXCT7E",
        "lat": 13.0500,
        "lon": 80.2824,
        "tags": ["Beach", "Crowded"],
        "hidden_percentile": 0.1,
        "road_access": RoadAccessLevel.PAVED,
        "safety_rating": SafetyRating.HIGH,
    },
    {
        "name": "Pichavaram Mangrove",
        "description": "One of the world's largest mangrove forests, best explored by boat.",
        "image_url": "https://unsplash.com/photos/green-trees-on-forest-during-daytime-qLTsA_plc1k",
        "lat": 11.4118,
        "lon": 79.7997,
        "tags": ["Boating", "Adventure"],
        "hidden_percentile": 0.6,
        "road_access": RoadAccessLevel.OFF_ROAD,
        "safety_rating": SafetyRating.MODERATE,
    },
]


async def seed() -> None:
    """Insert archetype places with dynamically computed weather history."""
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client.get_default_database("watrs_db")
    collection = db["places_live"]

    inserted = 0
    skipped = 0
    all_comforts: List[Dict[str, Any]] = []  # for verification table

    for arch in ARCHETYPES:
        # Duplicate check by name
        existing = await collection.find_one({"name": arch["name"]})
        if existing:
            logger.info("Skipping '%s' — already exists.", arch["name"])
            skipped += 1
            continue

        logger.info("Fetching weather history for '%s' …", arch["name"])
        comfort = await calculate_historical_comfort(arch["lat"], arch["lon"])

        place = Place(
            name=arch["name"],
            description=arch["description"],
            image_url=arch["image_url"],
            location=GeoJSONPoint(coordinates=(arch["lon"], arch["lat"])),
            watrs_tags=arch["tags"],
            safety_metadata=SafetyMetadata(
                road_access=arch["road_access"],
                safety_rating=arch["safety_rating"],
            ),
            metrics=PlaceMetrics(
                hidden_percentile=arch["hidden_percentile"],
                weather_comfort_history=comfort,
            ),
        )

        await collection.insert_one(place.to_mongo())
        inserted += 1
        logger.info("Inserted '%s' ✓", arch["name"])

        all_comforts.append({"name": arch["name"], "comfort": comfort})

    logger.info("Seed complete: %d inserted, %d skipped", inserted, skipped)

    # ── Verification table ──────────────────────────────────────────────
    if all_comforts:
        _print_verification_table(all_comforts)

    client.close()


def _print_verification_table(comforts: List[Dict[str, Any]]) -> None:
    """Print Jan / May / Oct comfort scores as a formatted table."""
    highlight_months = ["Jan", "May", "Oct"]

    print("\n" + "=" * 62)
    print("  COMFORT SCORE VERIFICATION (Jan / May / Oct)")
    print("=" * 62)
    print(f"  {'Place':<25} {'Jan':>8} {'May':>8} {'Oct':>8}")
    print("  " + "-" * 49)

    for entry in comforts:
        name = entry["name"]
        c = entry["comfort"]
        jan = c.get("Jan", "—")
        may = c.get("May", "—")
        oct_ = c.get("Oct", "—")
        print(f"  {name:<25} {jan:>8} {may:>8} {oct_:>8}")

    print("=" * 62)
    print("  Formula: Comfort = 0.5 × T_score + 0.5 × R_score")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    asyncio.run(seed())
