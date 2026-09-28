import os
from src.ingestion.legal_parser import ConstitutionalParser
from src.retrieval.hybrid_engine import HybridSearchEngine
from src.pipeline.crag_pipeline import ConstitutionalCRAG

def load_constitutional_data():
    """تحميل دساتير الدول الأربع وتجزئتها"""
    sources = [
        {"path": "data/raw/egypt_constitution.pdf", "country": "Egypt", "lang": "ar"},
        {"path": "data/raw/us_constitution.pdf", "country": "US", "lang": "en"},
        {"path": "data/raw/germany_constitution.pdf", "country": "Germany", "lang": "de"},
        {"path": "data/raw/france_constitution.pdf", "country": "France", "lang": "fr"},
    ]

    all_docs = []
    for src in sources:
        if os.path.exists(src["path"]):
            docs = ConstitutionalParser.load_pdf(src["path"], src["country"], src["lang"])
            all_docs.extend(docs)
            print(f"تم تحميل دستور {src['country']}: {len(docs)} مادة/وحدة.")

    # نصوص تجريبية في حال لم تكن ملفات الـ PDF متوفرة في المسار
    if not all_docs:
        print("تنبيه: ملفات الـ PDF غير موجودة في data/raw/. سيتم استخدام بيانات تجريبية.")
        all_docs = [
            ConstitutionalParser.load_pdf.__globals__["Document"](
                page_content="مادة 140: يُنتخب رئيس الجمهورية لمدة ست سنوات ميلادية، تبدأ من اليوم التالي لانتهاء مدة سلفه، ولا يجوز أن يتولى الرئاسة لأكثر من مدتين رئاسيتين متتاليتين.",
                metadata={"country": "Egypt", "article_number": "مادة 140", "language": "ar"}
            ),
            ConstitutionalParser.load_pdf.__globals__["Document"](
                page_content="Article II, Section 1: The executive Power shall be vested in a President of the United States of America. He shall hold his Office during the Term of four Years...",
                metadata={"country": "US", "article_number": "Article II Section 1", "language": "en"}
            )
        ]

    return all_docs

def main():
    print("1. جاري استيعاب النصوص وفهرستها...")
    docs = load_constitutional_data()

    print("2. بناء محرك البحث الهجين (Dense + BM25)...")
    search_engine = HybridSearchEngine(documents=docs)

    print("3. تهيئة خط سير عمل الـ CRAG...")
    crag_system = ConstitutionalCRAG(hybrid_engine=search_engine)

    # تجربة استعلام مقارنة دستورية متعددة اللغات
    test_query = "قارن بين مدة ولاية رئيس الجمهورية في كل من الدستور المصري والدستور الأمريكي مع ذكر أرقام المواد الدستورية المستند إليها."
    print(f"\nالاستعلام: {test_query}\n")
    print("4. المعالجة واستخراج الإجابة الموثقة:\n" + "=" * 50)
    
    answer = crag_system.run(test_query)
    print(answer)
    print("=" * 50)

if __name__ == "__main__":
    main()