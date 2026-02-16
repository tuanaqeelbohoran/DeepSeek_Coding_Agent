import pytest
from pytest_asyncio import fixture
from api.app import app, get_data
from httpx import ASGITransport, AsyncClient

@fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

@pytest.mark.asyncio
async def test_read_data(async_client):
    response = await async_client.get("/data")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from FastAPI!"}

@pytest.mark.asyncio
async def test_root(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI app"}
