from nltk import word_tokenize, sent_tokenize, pos_tag , WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import string
import re


def remove_tashkeel(text):
    # Removes diacritics (tashkeel) from Arabic text using regex.
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

def normalize_arabic(text: str, nlp, arabic_stopwords: set) -> str:
    #Cleans and normalizes Arabic text by removing diacritics, punctuation,stopwords, and converting words to their lemmas.
    if not text or not text.strip():
        return ""

    # Step 1: Remove initial diacritics
    text = remove_tashkeel(text)
    doc = nlp(text)
    cleaned_tokens = []
    
    # Step 2: Iterate over sentences and words
    for sentence in doc.sentences:
        for word in sentence.words:
            # Filter out punctuation, symbols, and numbers if enabled
            if word.upos in ['PUNCT', 'SYM', 'NUM']:
                continue

            raw = remove_tashkeel(word.text.strip())
            lemma = remove_tashkeel(word.lemma.strip()) if word.lemma else raw
            
            # Filter out stopwords if enabled
            if (raw in arabic_stopwords or lemma in arabic_stopwords or raw.isdigit() != 0 or  lemma.isdigit() != 0):
                continue
            
            cleaned_tokens.append(lemma)
            
    # Step 3: Join processed tokens into a single cleaned string
    return " ".join(cleaned_tokens)

def get_pos(tag: str):
    """Maps NLTK POS tags to WordNet POS tags."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    return wordnet.NOUN

def normalize_english(text: str, lemmatizer, punctuations: set, english_StopWords: set) -> str:
    """Cleans and normalizes English text with POS tagging and Lemmatization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)   # Remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)  # Keep alphabet letters only
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    
    # Punctuations and numbers are already stripped by regex above
    new_tokens = [
        lemmatizer.lemmatize(word, get_pos(tag)) 
        for word, tag in tags 
        if word not in english_StopWords
    ]
        
    return " ".join(new_tokens)

def clean_general_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()