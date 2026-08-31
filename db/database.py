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
    _add_missing_columns()


# SQLModel's create_all only creates tables that do not exist; it never alters
# one that does. A new column therefore appears in the model and not in an
# existing database, and every query against it fails on a machine that has
# been running for a while -- which is precisely the machine you least want to
# break. Add the columns explicitly, ignoring the ones already present.
_ADDED_COLUMNS = [
    ("inferencelog", "requested_model", "TEXT"),
]


def _add_missing_columns():
    if not db_url.startswith("sqlite"):
        return  # other backends should use a real migration tool
    from sqlalchemy import text
    with engine.connect() as conn:
        for table, column, coltype in _ADDED_COLUMNS:
            existing = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})"))}
            if not existing:
                continue  # table not created yet; create_all will have made it
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
                conn.commit()

def get_session():
    with Session(engine) as session:
        yield session
