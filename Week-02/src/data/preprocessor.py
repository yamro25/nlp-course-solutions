import re
from collections import Counter
import pickle

# قاموس فك الاختصارات الشائعة للحفاظ على سياق النفي والضمائر
CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "i'm": "i am", "you're": "you are", "they're": "they are",
    "it's": "it is", "that's": "that is", "what's": "what is",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not"
}

def clean_text(text: str) -> list:
    """
    Cleans toxic text specifically: expands contractions, handles special Wikipedia patterns,
    preserves punctuation sentiment (exclamation/question), reduces character repetition,
    and returns tokenized words.
    """
    text = str(text).lower()

    # 1. إزالة روابط الويب وعناوين الـ IP وروابط ويكيبيديا الهيكلية
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " ", text)
    text = re.sub(r"redirect talk:\S+", " ", text)
    text = re.sub(r"\[\[.*?\]\]", " ", text)

    # 2. فك الاختصارات (لتجنب حذف كلمات النفي المهمة)
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expanded, text)

    # 3. تحويل علامات الترقيم الدالة على الغضب والاستنكار إلى توكنز دلالية
    text = re.sub(r"!+", " tag_exclam ", text)
    text = re.sub(r"\?+", " tag_question ", text)

    # 4. تقليص التكرار المفرط للحروف (مثل: noooo -> noo أو shiiiiit -> shiit)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 5. تنظيف باقي الرموز غير الحرفية (مع الحفاظ على التوكنز الخاصة التي تبدأ بـ tag_)
    text = re.sub(r"[^a-zA-Z_]", " ", text)

    # 6. التقطيع واستبعاد الحروف الفردية غير المفيدة
    tokens = [w for w in text.split() if len(w) > 1 or w in ['a', 'i']]
    return tokens

def build_vocab(tokenized_texts: list, max_vocab_size: int) -> dict:
    """Builds word-to-index vocabulary."""
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab

def encode_and_pad(tokens: list, vocab: dict, max_len: int) -> list:
    """Maps tokens to indices and pads/truncates to max_len."""
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

def save_vocab(vocab: dict, filepath: str):
    """Saves vocabulary dictionary to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(filepath: str) -> dict:
    """Loads vocabulary dictionary from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)