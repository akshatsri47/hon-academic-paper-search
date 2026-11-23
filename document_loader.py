
import csv
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=500,
            chunk_overlap=50,
        )

    def preprocess(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def load_documents(self) -> list[Document]:
        documents = []
        seen_ids = set()

        with self.file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            fields = reader.fieldnames or []
            full_text_field = "full_text" if "full_text" in fields else "fulltext"

            required = {
                "paper_id", "title", "abstract", full_text_field,
                "keywords", "year", "discipline"
            }

            if not required.issubset(fields):
                raise ValueError("Missing required CSV columns")

            for row in reader:
                paper_id = self.preprocess(row["paper_id"])

                if not paper_id:
                    raise ValueError("Paper ID cannot be empty")

                if paper_id in seen_ids:
                    raise ValueError(f"Duplicate paper ID: {paper_id}")

                seen_ids.add(paper_id)

                title = self.preprocess(row["title"])
                abstract = self.preprocess(row["abstract"])
                fulltext = self.preprocess(row[full_text_field])
                keywords = self.preprocess(row["keywords"])
                discipline = self.preprocess(row["discipline"])

                try:
                    year = int(row["year"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid publication year for {paper_id}"
                    )

                content = (
                    f"Title: {title}\n"
                    f"Abstract: {abstract}\n"
                    f"Keywords: {keywords}\n"
                    f"Discipline: {discipline}\n"
                    f"Full Text: {fulltext}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "paper_id": paper_id,
                            "title": title,
                            "keywords": keywords,
                            "year": year,
                            "discipline": discipline,
                        },
                    )
                )

        if not documents:
            raise ValueError("No research papers found")

        return documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        return self.splitter.split_documents(documents)
