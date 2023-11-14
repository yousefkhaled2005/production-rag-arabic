from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import QueryRequest, QueryResponse, IndexBatchRequest
from src.retriever.bm25_retriever import ArabicBM25Retriever
from src.retriever.hybrid import ReciprocalRankFusion
from src.generator.qa_pipeline import ArabicRAGPipeline
from src.config import settings
import logging

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-Throughput Enterprise RAG Microservice for Arabic Legal Information Retrieval."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory pipeline initialisation for standalone testability
bm25 = ArabicBM25Retriever()
rrf = ReciprocalRankFusion(k_constant=settings.RRF_K_CONSTANT)


class MockRetriever:
    def __init__(self, bm25_retriever):
        self.bm25 = bm25_retriever

    def retrieve(self, query: str, top_k: int = 5):
        return self.bm25.search(query, top_k=top_k)


pipeline = ArabicRAGPipeline(hybrid_retriever=MockRetriever(bm25))


@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@app.post("/api/v1/documents/index", tags=["Ingestion"])
async def index_documents(request: IndexBatchRequest):
    try:
        chunks = [doc.dict() | {"chunk_id": f"{doc.document_id}_{idx}"} for idx, doc in enumerate(request.documents)]
        bm25.index(chunks)
        return {"status": "success", "indexed_count": len(chunks)}
    except Exception as e:
        logger.error(f"Failed to index documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_legal_rag(request: QueryRequest):
    try:
        result = pipeline.answer_query(request.query, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
