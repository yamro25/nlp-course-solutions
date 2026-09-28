from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from src.core.factories import GenericFactory
from src.core.config import app_config

class HybridSearchEngine:
    def __init__(self, documents: List[Document]):
        embeddings = GenericFactory.get_embeddings()
        retrieval_cfg = app_config.retrieval
        vdb_cfg = app_config.vector_db

        # 1. الاسترجاع الدلالي (Dense Vector Search)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vdb_cfg.persist_directory
        )
        self.dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": retrieval_cfg.top_k}
        )

        # 2. الاسترجاع اللفظي الحرفي (Sparse BM25 Search)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = retrieval_cfg.top_k

    def invoke(self, query: str) -> List[Document]:
        """دمج نتائج البحثين (Dense + Sparse) باستخدام خوارزمية Reciprocal Rank Fusion (RRF)"""
        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # RRF Scoring (k=60 هو المعيار القياسي)
        rrf_k = 60
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + (1.0 / (rank + rrf_k))

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + (1.0 / (rank + rrf_k))

        # ترتيب المستندات حسب النقاط الإجمالية
        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)
        top_k = app_config.retrieval.top_k
        return [doc_map[content] for content in sorted_contents[:top_k]]

    def get_retriever(self):
        """إرجاع الكائن نفسه ليتوافق مع invoke() في الـ pipeline"""
        return self