"""
Locust Load Testing Suite for Arabic Legal RAG Microservice.
Simulates concurrent user queries against FastAPI endpoints and reports
P50, P90, P95, P99 latency percentiles and token throughput.
"""
from locust import HttpUser, task, between
import random

SAMPLE_QUERIES = [
    "ما هي حالات فصل العامل لخطأ جسيم وفق المادة 69 من قانون العمل؟",
    "متى ينتهي عقد العمل محدد المدة وما هي شروط التعويض؟",
    "ما هي مهلة الإخطار الواجبة لإنهاء عقد العمل غير محدد المدة؟",
    "كيف يتم حساب مكافأة نهاية الخدمة للعامل عند بلوغ سن الستين؟",
    "ما هي حقوق العامل في حال إنهاء العقد تعسفياً من جانب صاحب العمل؟"
]


class LegalRAGLoadTester(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def query_rag_pipeline(self):
        payload = {
            "query": random.choice(SAMPLE_QUERIES),
            "top_k": 5,
            "use_reranker": True,
            "temperature": 0.1
        }
        with self.client.post("/api/v1/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data and len(data.get("citations", [])) > 0:
                    response.success()
                else:
                    response.failure("Response missing citations or answer")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def check_health(self):
        self.client.get("/health")
