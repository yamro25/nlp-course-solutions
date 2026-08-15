import re
import string

# Arabic Normalization & Cleaning
ARABIC_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
ARABIC_TATWEEL = re.compile(r'\u0640')

def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub('', text)
    text = ARABIC_TATWEEL.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_english(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text) # Remove non-alphabetical chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_general_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()