#import joblib
import sys
import os
import gradio as gr
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.Arabic_model import ArabicSentimentModel
from src.English_model import EnglishSentimentModel
from langdetect import detect

class MultilingualSentimentPipeline:
    def __init__(self):
        """base_dir = os.path.dirname(os.path.abspath(__file__))
        lang_model_path = os.path.join(base_dir, 'src', 'Language_classifier_weights.pkl') 
        
        print("Loading models...")
        self.lang_model = joblib.load(lang_model_path) """

        self.arabic_sentiment = ArabicSentimentModel()
        self.english_sentiment = EnglishSentimentModel() 

    def process(self, user_text):
        if not user_text.strip():
            return "برجاء إدخال نص", "غير محدد"
            
        # 1. Language Detection
        #detected_lang = self.lang_model.predict([user_text])[0]
        
        try:
            detected_lang = "Arabic" if detect(user_text) == "ar" else "English"
        except Exception:
            detected_lang = "English"
        
        # 2. Route to Sentiment Analysis
        if detected_lang == 'Arabic':
            sentiment = self.arabic_sentiment.predict(user_text)
        else:
            sentiment = self.english_sentiment.predict(user_text)
            
        return detected_lang, sentiment

# تشغيل الواجهة التفاعلية
if __name__ == "__main__":
    pipeline = MultilingualSentimentPipeline()

    def analyze_interface(text):
        lang, sentiment = pipeline.process(text)
        return lang, sentiment

    # بناء واجهة Gradio
    demo = gr.Interface(
        fn=analyze_interface,
        inputs=gr.Textbox(lines=3, placeholder="أدخل النص هنا للتجربة / Enter text here..."),
        outputs=[
            gr.Textbox(label="اللغة المكتشفة (Detected Language)"),
            gr.Textbox(label="تحليل المشاعر (Sentiment Classification)")
        ],
        title="نظام تحليل المشاعر متعدد اللغات",
        description="يقوم النظام بتحديد لغة النص تلقائياً (عربي / إنجليزي) ثم توجيهه للموديل المناسب لتحديد المشاعر.",
        examples=[
            ["التطبيق ممتاز جداً وسهل الاستخدام وبدون مشاكل"],
            ["The movie was absolutely terrible and boring, wasted my time."],
            ["خدمة العملاء سيئة للغاية وغير متعاونين"],
            ["It was a brilliant performance with great visuals and amazing story!"]
        ]
    )

    demo.launch()