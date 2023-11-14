from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_index_and_query():
    # Index test document
    doc_payload = {
        "documents": [
            {
                "document_id": "test_law",
                "text": "ينتهي عقد العمل بانتهاء مدته المحددة دون حاجة إلى إنذار مسبق.",
                "metadata": {"type": "labor"}
            }
        ]
    }
    index_res = client.post("/api/v1/documents/index", json=doc_payload)
    assert index_res.status_code == 200

    # Query
    query_payload = {
        "query": "متى ينتهي عقد العمل؟",
        "top_k": 3
    }
    query_res = client.post("/api/v1/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert "answer" in data
    assert len(data["citations"]) >= 1
