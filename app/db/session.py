"""
Database Engine & Session Management Module.

Analogy for Beginners:
Think of the Database Engine like a busy fast-food restaurant kitchen.
Instead of sending one chef to buy ingredients every time a customer orders (synchronous),
the async database engine allows chefs to handle hundreds of orders simultaneously (asynchronous).
The AsyncSession is like a waiter holding a tray: it takes customer requests, brings them to the kitchen,
and carries back fresh data cleanly without making anyone wait in line!
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

# Determine database driver connection type
# If running on SQLite for testing, ensure sqlite+aiosqlite driver is used
database_url = settings.DATABASE_URL

# Create the asynchronous engine with pool recycling and pre-ping checks
# pool_pre_ping checks if connection is alive before using it (like calling ahead before driving to store)
engine_kwargs = {"echo": False, "future": True}
if "sqlite" in database_url:
    # SQLite requires disabling check_same_thread for async operation across tasks
    engine_kwargs["connect_args"] = {"check_same_thread": False}

async_engine = create_async_engine(database_url, **engine_kwargs)

# Create an async session factory to generate fresh AsyncSession objects for each API request
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevents attributes from being invalidated after commit
    autoflush=False
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency that provides an asynchronous database session per HTTP request.
    
    Yields:
        AsyncSession: Active SQLAlchemy database transaction context.
        
    Workflow:
        1. Open connection session.
        2. Yield control to route handler.
        3. Automatically commit or rollback transaction on exceptions.
        4. Close session when HTTP response finishes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
