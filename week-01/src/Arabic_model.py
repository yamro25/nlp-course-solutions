import joblib
import os
from Preprocessing_pipeline import normalize_arabic
import stanza
import nltk
from nltk.corpus import stopwords

# 1. Download models and resources
stanza.download('ar')
nltk.download('stopwords', quiet=True)

# Initialize Stanza and Stopwords
nlp = stanza.Pipeline('ar', processors='tokenize,mwt,pos,lemma', verbose=False)
arabic_stopwords = set(stopwords.words('arabic'))

class ArabicSentimentModel:
    def __init__(self, model_path='Arabic_model_weights.pkl'):
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
        self.model = joblib.load(model_path)

    def predict(self, text):
        clean_text = normalize_arabic(text, nlp, arabic_stopwords)
        prediction = self.model.predict([clean_text])[0]
        return prediction