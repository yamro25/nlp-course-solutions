import joblib
import os
from src.Preprocessing_pipeline import normalize_english

class EnglishSentimentModel:
    def __init__(self, model_path='English_model_weights.pkl'):
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
        self.model = joblib.load(model_path)

    def predict(self, text):
        clean_text = normalize_english(text)
        prediction = self.model.predict([clean_text])[0]
        return prediction