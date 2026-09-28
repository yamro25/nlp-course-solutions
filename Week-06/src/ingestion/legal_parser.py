import os
import re
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document

class ConstitutionalParser:
    """محلل مخصص لاستخراج المواد والبيانات الوصفية (Metadata) لكل دولة"""

    ARTICLE_PATTERNS = {
        "Egypt": r"(مادة\s*[\(\[\{]?\s*\d+\s*[\)\]\}]?)",
        "US": r"(Article\s+[IVXLCDM]+|Amendment\s+[IVXLCDM]+)",
        "Germany": r"(Artikel\s+\d+[a-z]?)",
        "France": r"(Article\s+\d+)"
    }

    @classmethod
    def load_pdf(cls, file_path: str, country: str, language: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")

        reader = PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        pattern = cls.ARTICLE_PATTERNS.get(country)
        docs: List[Document] = []

        if not pattern:
            docs.append(Document(
                page_content=full_text,
                metadata={"country": country, "language": language, "source": file_path}
            ))
            return docs

        parts = re.split(pattern, full_text)
        # دمج عنوان المادة مع نصها
        for i in range(1, len(parts), 2):
            article_num = parts[i].strip()
            article_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            clean_content = f"{article_num}: {article_body}"

            docs.append(
                Document(
                    page_content=clean_content,
                    metadata={
                        "country": country,
                        "language": language,
                        "article_number": article_num,
                        "source": file_path
                    }
                )
            )
        return docs