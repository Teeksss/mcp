import pytest
from src.services.routing.smart_router import SmartRouter, EndpointUnavailable

@pytest.fixture
def sample_endpoints():
    return {
        "gpt-4": [
            {"url": "http://ep1", "health": True, "load": 0.3},
            {"url": "http://ep2", "health": False, "load": 0.1},
            {"url": "http://ep3", "health": True, "load": 0.1},
        ],
        "llama": [
            {"url": "http://llama1", "health": False, "load": 0.5}
        ]
    }

def test_route_returns_lowest_load(sample_endpoints):
    router = SmartRouter(sample_endpoints)
    url = router.route("gpt-4")
    assert url in ["http://ep3"]

def test_route_raises_if_no_healthy():
    router = SmartRouter({"llama": [{"url": "http://x", "health": False, "load": 0.2}]})
    with pytest.raises(EndpointUnavailable):
        router.route("llama")

def test_multiple_candidates_random_choice(monkeypatch, sample_endpoints):
    # Her iki endpoint'in de yükü eşit olursa rastgele birini seçer
    endpoints = {
        "gpt-4": [
            {"url": "http://ep1", "health": True, "load": 0.2},
            {"url": "http://ep2", "health": True, "load": 0.2},
        ]
    }
    router = SmartRouter(endpoints)
    # Rastgeleliği sabitle
    monkeypatch.setattr("random.choice", lambda eps: eps[0])
    url = router.route("gpt-4")
    assert url == "http://ep1"