
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from document_loader import DocumentLoader
from vector_store import ResearchVectorStore


class ResearchSearch:
    def __init__(
        self,
        file_path: str,
        recency_weight: float = 0.2,
    ):
        if not 0 <= recency_weight <= 1:
            raise ValueError("Recency weight must be between 0 and 1")

        self.loader = DocumentLoader(file_path)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.vector_store = ResearchVectorStore(self.embeddings)
        self.recency_weight = recency_weight

        self.paper_count = 0
        self.min_year = 0
        self.max_year = 0

    def setup(self) -> None:
        documents = self.loader.load_documents()

        self.paper_count = len(documents)
        years = [doc.metadata["year"] for doc in documents]

        self.min_year = min(years)
        self.max_year = max(years)

        chunks = self.loader.split_documents(documents)
        self.vector_store.build(chunks)

        print(
            f"Indexed {self.paper_count} papers "
            f"into {len(chunks)} chunks"
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        if self.paper_count == 0:
            raise RuntimeError("Run setup() before searching")

        matches = self.vector_store.search(
            query=query,
            k=self.vector_store.get_total_chunks(),
        )

        papers = {}

        # Keep the best matching chunk for each paper.
        for document, distance in matches:
            paper_id = document.metadata["paper_id"]

            if (
                paper_id not in papers
                or distance < papers[paper_id]["distance"]
            ):
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": document.metadata["title"],
                    "keywords": document.metadata["keywords"],
                    "year": document.metadata["year"],
                    "discipline": document.metadata["discipline"],
                    "distance": float(distance),
                }

        # Semantic ranking: smaller FAISS distance is better.
        candidates = sorted(
            papers.values(),
            key=lambda paper: paper["distance"],
        )

        total = len(candidates)

        for i, paper in enumerate(candidates):
            semantic_score = (
                1.0 if total == 1
                else 1 - i / (total - 1)
            )

            recency_score = (
                (paper["year"] - self.min_year)
                / (self.max_year - self.min_year)
                if self.max_year > self.min_year
                else 1.0
            )

            paper["semantic_score"] = semantic_score
            paper["recency_score"] = recency_score

            paper["final_score"] = (
                (1 - self.recency_weight) * semantic_score
                + self.recency_weight * recency_score
            )

        ranked = sorted(
            candidates,
            key=lambda paper: (
                -paper["final_score"],
                paper["distance"],
            ),
        )

        # Select by relevance and recency, then display the chosen papers newest first.
        # Stable sorting preserves score order for papers published in the same year.
        selected = ranked[:k]
        return sorted(selected, key=lambda paper: paper["year"], reverse=True)


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env")

    engine = ResearchSearch(
        file_path="research_papers.csv",
        recency_weight=0.2,
    )

    engine.setup()

    while True:
        query = input("\nEnter search query (exit to quit): ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            results = engine.search(query, k=3)
        except ValueError as error:
            print(f"Search error: {error}")
            continue

        if not results:
            print("No matching papers found")
            continue

        for i, paper in enumerate(results, start=1):
            print(f"\n{i}. {paper['title']}")
            print(f"Paper ID: {paper['paper_id']}")
            print(f"Year: {paper['year']}")
            print(f"Discipline: {paper['discipline']}")
            print(f"Keywords: {paper['keywords']}")
            print(f"Semantic Score: {paper['semantic_score']:.3f}")
            print(f"Recency Score: {paper['recency_score']:.3f}")
            print(f"Final Score: {paper['final_score']:.3f}")


if __name__ == "__main__":
    main()
