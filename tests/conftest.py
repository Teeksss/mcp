import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.session import Base
from src.main import app
from src.config.settings import settings

# Test database
TEST_DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token"}