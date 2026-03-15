"""
scripts/recommend_coimbatore.py
───────────────────────────────
Find and print recommendations for a user in Coimbatore for Jan & Feb.

Filters:
  1.  weather_comfort_history.Jan > 0.8  OR  .Feb > 0.8
  2.  Prioritise hidden gems  (metrics.hidden_percentile > 0.7)
  3.  Sort by distance from Coimbatore (uses $geoNear)

Output:
  Formatted table with Place, Month, Comfort, and reason tag.

Usage:
    .venv\\Scripts\\python.exe scripts/recommend_coimbatore.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from motor.motor_asyncio import AsyncIOMotorClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import get_settings

settings = get_settings()

# ── Coimbatore coordinates ──────────────────────────────────────────────────
CBE_LAT = 11.0168
CBE_LON = 76.9558


async def recommend() -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client.get_default_database("watrs_db")
    collection = db["places_live"]

    # ── Stage 1: $geoNear from Coimbatore, generous 500 km radius ───────
    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [CBE_LON, CBE_LAT]},
                "distanceField": "dist_meters",
                "maxDistance": 500_000,        # 500 km in meters
                "spherical": True,
            }
        },
        # Stage 2: filter Jan > 0.8 OR Feb > 0.8
        {
            "$match": {
                "$or": [
                    {"metrics.weather_comfort_history.Jan": {"$gt": 0.8}},
                    {"metrics.weather_comfort_history.Feb": {"$gt": 0.8}},
                ]
            }
        },
        # Stage 3: add computed fields
        {
            "$addFields": {
                "is_hidden_gem": {
                    "$gt": ["$metrics.hidden_percentile", 0.7]
                },
                "dist_km": {
                    "$round": [{"$divide": ["$dist_meters", 1000]}, 1]
                },
            }
        },
        # Stage 4: sort — hidden gems first, then by distance
        {"$sort": {"is_hidden_gem": -1, "dist_meters": 1}},
    ]

    results: List[Dict[str, Any]] = await collection.aggregate(pipeline).to_list(length=50)

    if not results:
        print("\nNo places match the criteria.\n")
        client.close()
        return

    # ── Build table rows ────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    for doc in results:
        name = doc["name"]
        comfort = doc.get("metrics", {}).get("weather_comfort_history", {})
        jan = comfort.get("Jan")
        feb = comfort.get("Feb")
        hidden = doc.get("metrics", {}).get("hidden_percentile", 0)
        dist = doc.get("dist_km", "?")
        road = doc.get("safety_metadata", {}).get("road_access", "unknown")

        # Build reason tags
        reasons = []
        if hidden > 0.7:
            reasons.append("Hidden Gem 💎")
        if jan and jan >= 0.9:
            reasons.append("Excellent Jan Weather")
        elif jan and jan > 0.8:
            reasons.append("High Jan Comfort")
        if feb and feb >= 0.9:
            reasons.append("Excellent Feb Weather")
        elif feb and feb > 0.8:
            reasons.append("High Feb Comfort")
        if hidden <= 0.3:
            reasons.append("Low Crowds")
        if road == "off_road":
            reasons.append("⚠️ Off-Road Access")

        # One row per qualifying month
        for month, score in [("Jan", jan), ("Feb", feb)]:
            if score and score > 0.8:
                rows.append({
                    "name": name,
                    "month": month,
                    "comfort": score,
                    "dist": dist,
                    "reason": " + ".join(reasons) if reasons else "Good Comfort",
                })

    # ── Print formatted table ───────────────────────────────────────────
    print()
    print("=" * 95)
    print("   📍 COIMBATORE — WINTER RECOMMENDATIONS (Jan & Feb, Comfort > 0.8)")
    print("=" * 95)
    print(f"   {'Place':<22} {'Month':>5} {'Comfort':>9} {'Dist (km)':>10}   {'Why Recommended'}")
    print("   " + "─" * 88)

    for r in rows:
        c_str = f"{r['comfort']:.2f}"
        print(f"   {r['name']:<22} {r['month']:>5} {c_str:>9} {r['dist']:>10}   {r['reason']}")

    print("=" * 95)
    print(f"   Total matches: {len(rows)} rows from {len(results)} places")
    print("=" * 95)
    print()

    client.close()


if __name__ == "__main__":
    asyncio.run(recommend())
