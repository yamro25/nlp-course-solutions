import joblib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.Arabic_model import ArabicSentimentModel
from src.English_model import EnglishSentimentModel

class MultilingualSentimentPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print("baseDir=", base_dir)
        lang_model_path = os.path.join(base_dir, 'src\Language_classifier_weights.pkl')
        
        print("Loading models...")
        self.lang_model = joblib.load(lang_model_path)
        self.arabic_sentiment = ArabicSentimentModel()
        self.english_sentiment = EnglishSentimentModel()

    def process(self, user_text):
        # Step 1: Language Detection
        detected_lang = self.lang_model.predict([user_text])[0]
        
        # Step 2: Route to Sentiment Analysis
        if detected_lang == 'Arabic':
            sentiment = self.arabic_sentiment.predict(user_text)
        else:
            sentiment = self.english_sentiment.predict(user_text)
            
        # Step 3: Format Output
        return {
            "User Text": user_text,
            "Language": detected_lang,
            "Sentiment Classification": sentiment
        }

if __name__ == "__main__":
    pipeline = MultilingualSentimentPipeline()
    
    # Test cases
    test_inputs = [
        "التطبيق ممتاز جداً وسهل الاستخدام وبدون مشاكل",
        "The movie was absolutely terrible and boring, wasted my time.",
        "خدمة العملاء سيئة للغاية وغير متعاونين",
        "It was a brilliant performance with great visuals and amazing story!"
    ]
    
    print("\n================ Pipeline Results ================\n")
    for text in test_inputs:
        result = pipeline.process(text)
        print(f"User Text               : {result['User Text']}")
        print(f"Language                : {result['Language']}")
        print(f"Sentiment Classification: {result['Sentiment Classification']}")
        print("-" * 50)