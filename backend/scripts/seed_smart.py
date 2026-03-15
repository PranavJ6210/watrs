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
    if 20.0 <= avg_max_temp <= 30.0: return 1.0
    elif 30.0 < avg_max_temp <= 33.0: return 0.85
    elif 33.0 < avg_max_temp <= 36.0: return 0.65
    elif 36.0 < avg_max_temp <= 39.0: return 0.45
    elif avg_max_temp > 39.0: return 0.25
    else: return 0.7  # < 20°C

def _rain_score(total_rain_mm: float) -> float:
    if total_rain_mm <= 40.0: return 1.0
    elif total_rain_mm <= 100.0: return 0.8
    elif total_rain_mm <= 200.0: return 0.6
    elif total_rain_mm <= 350.0: return 0.4
    else: return 0.2

async def calculate_historical_comfort(lat: float, lon: float) -> Dict[str, float]:
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

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_max"]
    rains = data["daily"]["rain_sum"]

    m_temps = {m: [] for m in range(1, 13)}
    m_rains = {m: [] for m in range(1, 13)}

    for i, d_str in enumerate(dates):
        m = int(d_str.split("-")[1])
        if temps[i] is not None: m_temps[m].append(temps[i])
        if rains[i] is not None: m_rains[m].append(rains[i])

    comfort = {}
    for m in range(1, 13):
        avg_t = sum(m_temps[m]) / len(m_temps[m]) if m_temps[m] else 30.0
        tot_r = sum(m_rains[m])
        comfort[MONTH_ABBR[m - 1]] = round(0.5 * _temp_score(avg_t) + 0.5 * _rain_score(tot_r), 4)
    return comfort

# ═══════════════════════════════════════════════════════════════════════════
# New Seed Spots (10 Unique Destinations)
# ════════════════════════════════════════───────────────────────────────────

ARCHETYPES: List[Dict[str, Any]] = [
    {"name": "Meghamalai", "description": "The High Wavy Mountains; a misty tea garden paradise.", "lat": 9.6922, "lon": 77.4080, "tags": ["Nature", "Tea Estates"], "hidden_percentile": 0.95, "road_access": RoadAccessLevel.OFF_ROAD, "safety_rating": SafetyRating.MODERATE},
    {"name": "Javadi Hills", "description": "Unexplored Eastern Ghats range with waterfalls and observatories.", "lat": 12.5833, "lon": 78.9167, "tags": ["Nature", "Adventure"], "hidden_percentile": 0.88, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.MODERATE},
    {"name": "Pollachi", "description": "The 'Coconut Capital' of TN, gateway to Anaimalai wildlife.", "lat": 10.6620, "lon": 77.0065, "tags": ["Nature", "Wildlife"], "hidden_percentile": 0.45, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Sirumalai", "description": "Peaceful retreat known for its herbal air and diverse flora.", "lat": 10.2000, "lon": 78.0000, "tags": ["Nature", "Relaxation"], "hidden_percentile": 0.82, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Yelagiri", "description": "A quaint hill station with boating lakes and rose gardens.", "lat": 12.5786, "lon": 78.6389, "tags": ["Hills", "Boating"], "hidden_percentile": 0.35, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Kotagiri", "description": "The oldest and quietest of the three Nilgiri hill stations.", "lat": 11.4333, "lon": 76.8833, "tags": ["Tea Estates", "Nature"], "hidden_percentile": 0.70, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Manjolai", "description": "Deep forest tea retreat located atop the Manimuthar Dam.", "lat": 8.5833, "lon": 77.3833, "tags": ["Tea Estates", "Trekking"], "hidden_percentile": 0.92, "road_access": RoadAccessLevel.OFF_ROAD, "safety_rating": SafetyRating.MODERATE},
    {"name": "Coonoor", "description": "Famous for Nilgiri tea, valleys, and colonial charm.", "lat": 11.3530, "lon": 76.7959, "tags": ["Hills", "Tea Estates"], "hidden_percentile": 0.25, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Theni", "description": "Known as the Earth's Hidden Paradise, surrounded by mountains.", "lat": 10.0150, "lon": 77.4830, "tags": ["Nature", "Agriculture"], "hidden_percentile": 0.60, "road_access": RoadAccessLevel.PAVED, "safety_rating": SafetyRating.HIGH},
    {"name": "Dhanushkodi", "description": "The hauntingly beautiful ghost town where oceans meet.", "lat": 9.1523, "lon": 79.4340, "tags": ["History", "Beach"], "hidden_percentile": 0.55, "road_access": RoadAccessLevel.OFF_ROAD, "safety_rating": SafetyRating.MODERATE},
]

# ═══════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════

async def seed() -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client.get_default_database("watrs_db")
    collection = db["places_live"]

    inserted, skipped = 0, 0
    all_comforts = []

    for arch in ARCHETYPES:
        existing = await collection.find_one({"name": arch["name"]})
        if existing:
            logger.info("Skipping '%s' — already exists.", arch["name"])
            skipped += 1
            continue

        logger.info("Fetching weather history for '%s' …", arch["name"])
        try:
            comfort = await calculate_historical_comfort(arch["lat"], arch["lon"])
            place = Place(
                name=arch["name"],
                description=arch["description"],
                image_url="https://placeholder.com/600",
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
            all_comforts.append({"name": arch["name"], "comfort": comfort})
            logger.info("Inserted '%s' ✓", arch["name"])
        except Exception as e:
            logger.error("Failed to seed '%s': %s", arch["name"], e)

    logger.info("Seed complete: %d inserted, %d skipped", inserted, skipped)
    if all_comforts: _print_verification_table(all_comforts)
    client.close()

def _print_verification_table(comforts: List[Dict[str, Any]]) -> None:
    # UPDATED: Includes Jan, Feb, May, and Oct columns
    print("\n" + "=" * 75)
    print("   COMFORT SCORE VERIFICATION (Jan / Feb / May / Oct)")
    print("=" * 75)
    print(f"   {'Place':<25} {'Jan':>10} {'Feb':>10} {'May':>10} {'Oct':>10}")
    print("   " + "-" * 67)
    for entry in comforts:
        c = entry["comfort"]
        j, f, m, o = c.get("Jan","—"), c.get("Feb","—"), c.get("May","—"), c.get("Oct","—")
        print(f"   {entry['name']:<25} {j:>10} {f:>10} {m:>10} {o:>10}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    asyncio.run(seed())