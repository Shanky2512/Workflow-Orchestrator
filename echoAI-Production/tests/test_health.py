
from fastapi.testclient import TestClient
from apps.gateway.main import app

def test_health():
    c = TestClient(app)
    r = c.get('/healthz')
    assert r.status_code == 200
