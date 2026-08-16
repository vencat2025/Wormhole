from sqlmodel import SQLModel, create_engine, Session
from config import settings

db_url = settings.DATABASE_URL

# Configure engine for PostgreSQL or SQLite
if db_url.startswith("postgresql") or db_url.startswith("postgres"):
    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )
else:
    # Default to SQLite with multi-thread check disabled
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False}
    )

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
