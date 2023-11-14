from typing import List, Dict, Optional
import time


ARABIC_LEGAL_SYSTEM_PROMPT = """أنت مساعد قانوني ذكي متخصص في القوانين والأنظمة العربية.
مهمتك تقديم إجابات دقيقة وموثوقة مستندة حصراً إلى النصوص والوثائق القانونية المرفقة في السياق.

القواعد الصارمة:
1. التزم بالنصوص القانونية المرفقة ولا تخترع أو تفترض أي أحكام لم تذكر فيها.
2. اذكر رقم المادة أو السند القانوني الوارد في السياق لكل نقطة تذكرها.
3. إذا لم تكن الإجابة موجودة بوضوح في المستندات المرفقة، قل صراحة: "المعلومات المتوفرة في المستندات لا تكفي للإجابة عن هذا السؤال".
4. الصياغة بلغة عربية فصحى قانونية سليمة وموجزة.
"""


class ArabicRAGPipeline:
    """End-to-End Orchestrator for Retrieval, Context Assembly, and Generation."""

    def __init__(self, hybrid_retriever, generator_model=None):
        self.retriever = hybrid_retriever
        self.generator_model = generator_model

    def answer_query(self, query: str, top_k: int = 5) -> Dict:
        start_time = time.time()

        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, top_k=top_k)

        # 2. Build context
        context_parts = []
        citations = []
        for chunk in chunks:
            context_parts.append(f"[{chunk['document_id']}]: {chunk['text']}")
            citations.append({
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk.get("rrf_score", chunk.get("score", 0.0)),
                "text": chunk["text"]
            })

        context_str = "\n\n".join(context_parts)

        # 3. Simulate generation / Call LLM
        # In production this calls vLLM / Ollama / OpenAI / LLaMA-3
        answer = self._synthesize_answer(query, context_str)
        latency_ms = (time.time() - start_time) * 1000

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "latency_ms": round(latency_ms, 2)
        }

    def _synthesize_answer(self, query: str, context: str) -> str:
        if not context:
            return "لم يتم العثور على وثائق قانونية ذات صلة بالسؤال المطروح."
        return (
            f"بناءً على المستندات القانونية المرفقة، وبشأن الاستفسار المتعلق بـ '{query}':\n\n"
            f"تفيد النصوص الواردة في المرجع بأن الأحكام والشروط تخضع للمحددات القانونية المنصوص عليها، "
            f"وتلزم الأطراف بمراعاة المواد النظامية الواجبة التطبيق."
        )
