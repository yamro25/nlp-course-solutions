import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# تحميل الإعدادات من ملف .env تلقائياً
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct")

st.set_page_config(
    page_title="فاحص النصوص الذكي | Text Analyzer",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ تحليل النصوص والسمية")
st.caption("كشف المحتوى السام (Toxic / Non-Toxic) وتحليل المشاعر (إيجابي / سلبي) عبر LangChain")

# تعريف بنية المخرجات المطلوبة لتشمل السمية والمشاعر معاً
class ContentAssessment(BaseModel):
    toxicity: str = Field(description="يجب أن تكون إما 'Toxic' أو 'Non-Toxic' فقط")
    sentiment: str = Field(description="يجب أن تكون إحدى القيم التالية فقط: 'Positive' أو 'Negative' أو 'Neutral'")
    explanation: str = Field(description="شرح مقتضب ومباشر لسبب هذا التقييم باللغة العربية")

def analyze_content(text: str) -> dict:
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,  # لضمان دقة واستقرار التصنيف
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Toxic and Sentiment Detector"
        }
    )

    parser = JsonOutputParser(pydantic_object=ContentAssessment)

    prompt = PromptTemplate(
        template=(
            "أنت نظام خبير في تحليل النصوص ومكافحة السمية وتحليل المشاعر.\n"
            "مهمتك فحص النص التالي واستخراج الآتي بدقة:\n"
            "1. هل النص سام ومسيء أم لا (Toxicity): اختر فقط 'Toxic' أو 'Non-Toxic'.\n"
            "2. ما هي نبرة ومشاعر النص (Sentiment): اختر فقط 'Positive' أو 'Negative' أو 'Neutral'.\n"
            "3. قدم تبريراً مختصراً باللغة العربية.\n\n"
            "{format_instructions}\n\n"
            "النص المراد تحليله:\n{text}\n"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain.invoke({"text": text})

# واجهة إدخال النص مباشرة دون أي أشرطة جانبية
input_text = st.text_area("أدخل النص المراد تحليله:", height=140, placeholder="اكتب أو الصق النص هنا...")

if st.button("تحليل النص 🚀", use_container_width=True):
    if not OPENROUTER_API_KEY:
        st.error("⚠️ لم يتم العثور على OPENROUTER_API_KEY داخل ملف .env.")
    elif not input_text.strip():
        st.warning("⚠️ يرجى إدخال نص أولاً.")
    else:
        with st.spinner("جاري تحليل النص وتقييم المشاعر..."):
            try:
                result = analyze_content(input_text)
                toxicity = result.get("toxicity", "غير محدد").strip()
                sentiment = result.get("sentiment", "غير محدد").strip()
                reason = result.get("explanation", "")

                st.markdown("---")
                
                # عرض النتيجة في أعمدة منسقة
                col1, col2 = st.columns(2)
                
                with col1:
                    if "Non-Toxic" in toxicity:
                        st.success(f"**فحص السمية:** {toxicity} ✅")
                    else:
                        st.error(f"**فحص السمية:** {toxicity} ⚠️")
                        
                with col2:
                    if "Positive" in sentiment:
                        st.success(f"**المشاعر:** إيجابي ({sentiment}) 😊")
                    elif "Negative" in sentiment:
                        st.warning(f"**المشاعر:** سلبي ({sentiment}) 😠")
                    else:
                        st.info(f"**المشاعر:** محايد ({sentiment}) 😐")

                st.info(f"**التفسير:** {reason}")

            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")