"""
core/database.py
────────────────
Database initialization utilities.

Called once during application startup (via the lifespan).
Ensures required indexes exist on critical collections.
"""

import logging
from typing import TYPE_CHECKING

import pymongo

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger("watrs.database")

# Collection name — single source of truth
PLACES_COLLECTION = "places_live"


async def initialize_db(db: "AsyncIOMotorDatabase") -> None:
    """
    Run one-time database bootstrapping:

    1. Ensure a ``2dsphere`` index on ``places_live.location`` for
       geospatial queries (``$near``, ``$geoWithin``, etc.).
    """
    collection = db[PLACES_COLLECTION]

    # ── 2dsphere index on location ──────────────────────────────────────
    index_name = "location_2dsphere"
    existing_indexes = await collection.index_information()

    if index_name in existing_indexes:
        logger.info(
            "Index '%s' already exists on '%s' — skipping creation.",
            index_name,
            PLACES_COLLECTION,
        )
    else:
        logger.info(
            "Creating 2dsphere index '%s' on '%s.location' …",
            index_name,
            PLACES_COLLECTION,
        )
        await collection.create_index(
            [("location", pymongo.GEOSPHERE)],
            name=index_name,
            background=True,
        )
        logger.info("Index '%s' created ✓", index_name)
