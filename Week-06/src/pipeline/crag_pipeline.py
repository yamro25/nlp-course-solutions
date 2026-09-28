from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.core.factories import GenericFactory
from src.retrieval.hybrid_engine import HybridSearchEngine
from src.pipeline.prompts import GRADER_PROMPT, QUERY_REWRITER_PROMPT, GENERATION_PROMPT

class ConstitutionalCRAG:
    """تنفيذ خط معالجة CRAG الحتمي باستخدام سلاسل LangChain LCEL بدون استدعاء Agent"""

    def __init__(self, hybrid_engine: HybridSearchEngine):
        self.retriever = hybrid_engine.get_retriever()
        self.llm = GenericFactory.get_llm()
        self.pipeline = self._build_chain()

    def _grade_and_correct(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """فحص الوثائق المسترجعة؛ إذا لم تكن ذات صلة، يتم تفعيل الاسترجاع التصحيحي"""
        docs: List[Document] = inputs["documents"]
        query: str = inputs["query"]

        context_repr = "\n\n".join([doc.page_content for doc in docs])
        grader_chain = GRADER_PROMPT | self.llm | StrOutputParser()
        score = grader_chain.invoke({"context": context_repr, "query": query}).strip().lower()

        if "yes" in score:
            return {"context_docs": docs, "query": query, "status": "direct_match"}

        # الخطوة التصحيحية: إعادة صياغة الاستعلام وإعادة البحث
        rewriter_chain = QUERY_REWRITER_PROMPT | self.llm | StrOutputParser()
        rewritten_query = rewriter_chain.invoke({"query": query}).strip()

        corrected_docs = self.retriever.invoke(rewritten_query)
        return {"context_docs": corrected_docs, "query": query, "status": "corrected_match"}

    def _format_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """تنسيق السياق مع البيانات الوصفية (Metadata)"""
        docs: List[Document] = inputs["context_docs"]
        chunks = []
        for doc in docs:
            country = doc.metadata.get("country", "غير محدد")
            art = doc.metadata.get("article_number", "مادة غير محددة")
            chunks.append(f"### [الدولة: {country} | المرجع: {art}]\n{doc.page_content}")

        return {
            "context": "\n\n".join(chunks) if chunks else "لا توجد نصوص دستورية مطابقة.",
            "query": inputs["query"]
        }

    def _build_chain(self):
        chain = (
            RunnablePassthrough.assign(
                documents=lambda x: self.retriever.invoke(x["query"])
            )
            | RunnableLambda(self._grade_and_correct)
            | RunnableLambda(self._format_context)
            | GENERATION_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    def run(self, query: str) -> str:
        return self.pipeline.invoke({"query": query})