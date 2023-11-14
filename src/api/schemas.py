from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class DocumentChunkInput(BaseModel):
    document_id: str = Field(..., example="egypt_labor_law_2003")
    text: str = Field(..., example="المادة 104: ينتهي عقد العمل محدد المدة بانقضاء مدته...")
    metadata: Optional[Dict[str, Any]] = None


class IndexBatchRequest(BaseModel):
    documents: List[DocumentChunkInput]


class QueryRequest(BaseModel):
    query: str = Field(..., example="ما هي أحكام عقد العمل محدد المدة؟")
    top_k: int = Field(5, ge=1, le=20)
    use_reranker: bool = True
    temperature: float = Field(0.1, ge=0.0, le=1.0)


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    latency_ms: float
