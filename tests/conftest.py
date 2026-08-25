"""
Pytest Async Fixtures & Test Client Configuration (`conftest.py`).

Analogy for Beginners:
Think of `conftest.py` like setting up a clean sandbox playground before every test game!
Instead of writing real data into a live customer database, pytest builds a temporary, in-memory database
inside computer RAM. When tests complete, the temporary database is wiped clean instantly without leaving a trace!
"""

import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.main import app
from app.db.session import get_async_db
from app.db.models import Base

# In-Memory SQLite Database URL for Lightning-Fast Isolation Testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


@pytest_asyncio.fixture(scope="function")
async def async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Fixture providing a fresh in-memory database session for each test function.
    Creates all ORM tables before the test runs and drops them afterwards.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestingSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(async_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture providing an HTTPX AsyncClient for sending virtual API HTTP requests to FastAPI routes.
    Overrides the global `get_async_db` dependency to use the isolated test database.
    """
    async def override_get_async_db():
        yield async_db

    app.dependency_overrides[get_async_db] = override_get_async_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

    app.dependency_overrides.clear()
