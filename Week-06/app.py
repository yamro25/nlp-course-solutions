import streamlit as st
import os
from src.ingestion.legal_parser import ConstitutionalParser
from src.retrieval.hybrid_engine import HybridSearchEngine
from src.pipeline.crag_pipeline import ConstitutionalCRAG

# إعدادات صفحة المتصفح
st.set_page_config(
    page_title="Constitutional RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص اتجاه النصوص ودعم العربية
st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stMarkdown, p, h1, h2, h3, h4, span { text-align: right; direction: rtl; }
    div[data-testid="stSidebar"] { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    """تحميل الوثائق وبناء الفهرس مرة واحدة فقط مع تخزينه في الكاش"""
    sources = [
        {"path": "data/raw/egypt_constitution.pdf", "country": "Egypt", "lang": "ar"},
        {"path": "data/raw/us_constitution.pdf", "country": "US", "lang": "en"},
        {"path": "data/raw/germany_constitution.pdf", "country": "Germany", "lang": "de"},
        {"path": "data/raw/france_constitution.pdf", "country": "France", "lang": "fr"},
    ]

    all_docs = []
    loaded_countries = []

    for src in sources:
        if os.path.exists(src["path"]):
            docs = ConstitutionalParser.load_pdf(src["path"], src["country"], src["lang"])
            all_docs.extend(docs)
            loaded_countries.append(src["country"])

    # Fallback بيانات تجريبية في حال عدم اكتمال الـ PDFs
    if not all_docs:
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
        loaded_countries = ["Egypt (Demo)", "US (Demo)"]

    search_engine = HybridSearchEngine(documents=all_docs)
    rag_system = ConstitutionalCRAG(hybrid_engine=search_engine)
    return rag_system, loaded_countries, len(all_docs)

# القائمة الجانبية (Sidebar)
with st.sidebar:
    st.title("⚙️ النظام والمعلومات")
    st.info("نظام RAG دستوري متعدد اللغات يغطي دساتير: مصر، أمريكا، ألمانيا، فرنسا.")
    
    with st.spinner("جاري تهيئة محرك البحث الهجين وقاعدة المتجهات..."):
        crag_system, countries, total_articles = initialize_system()

    st.success("تم تشغيل النظام بنجاح!")
    st.write(f"📊 **إجمالي المواد المفهرسة:** {total_articles}")
    st.write("🌍 **الدساتير المتاحة:**")
    for c in countries:
        st.write(f"- {c}")

    if st.button("🔄 مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# الشاشة الرئيسية
st.title("⚖️ المستشار الدستوري الذكي (Constitutional CRAG)")
st.caption("اطرح أي سؤال دستوري أو اطلب مقارنة بين الدساتير مع توثيق النصوص وأرقام المواد بدقة.")

# سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال المستخدم
if user_query := st.chat_input("اكتب استفسارك الدستوري هنا (مثال: قارن مدة ولاية الرئيس بين مصر وأمريكا)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في النصوص القانونية وتدقيق الإجابة..."):
            try:
                response = crag_system.run(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الاستعلام: {str(e)}")