"""
Main FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings
from api import auth, security
from core.dependencies import client

app = FastAPI(
    title="HyperLiquid Trading API",
    description="API for HyperLiquid autonomous trading system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(security.router)

@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection on startup."""
    app.mongodb_client = client
    app.mongodb = client[settings.MONGO_DB]

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown."""
    app.mongodb_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
