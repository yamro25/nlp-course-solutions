import joblib
import os
from src.Preprocessing_pipeline import normalize_english
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# 2. Setup preprocessing parameters
english_StopWords = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

class EnglishSentimentModel:
    def __init__(self, model_path='English_model_weights.pkl'):
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
        self.model = joblib.load(model_path)

    def predict(self, text):
        clean_text = normalize_english(text,lemmatizer, punctuations, english_StopWords)
        prediction = self.model.predict([clean_text])[0]
        return prediction