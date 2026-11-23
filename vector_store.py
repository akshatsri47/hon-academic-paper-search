
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


class ResearchVectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.store: FAISS | None = None

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents to index")

        self.store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

    def search(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        if self.store is None:
            raise RuntimeError("Build the FAISS index first")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        return self.store.similarity_search_with_score(
            query=query,
            k=min(k, self.get_total_chunks()),
        )

    def get_total_chunks(self) -> int:
        if self.store is None:
            return 0

        return int(self.store.index.ntotal)
