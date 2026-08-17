import re
import stanza

# 1. تحميل وتجهيز نموذج Stanza للغة العربية (مرة واحدة فقط)
stanza.download('ar', processors='tokenize,pos,lemma', verbose=False)
nlp = stanza.Pipeline('ar', processors='tokenize,pos,lemma', verbose=False)

# 2. دالة التنظيف (Preprocessing)
def clean_arabic_text(text):
    # إزالة التشكيل والتطويل
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'ـ+', '', text)
    
    # توحيد رسم الحروف (Normalization)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    
    # إزالة الروابط، الأرقام، وعلامات الترقيم
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # إزالة المساحات الزائدة
    return re.sub(r'\s+', ' ', text).strip()

# 3. النص والتنفيذ
text = "السَّلاَمُ عَلَيْكُمْ! يُشَارِكُ الطُّلَّابُ فِي الْمُسَابَقَاتِ لِتَطْوِيرِ الْمُسْتَشْفَيَاتِ."
cleaned_text = clean_arabic_text(text)

# 4. استخراج الـ Lemma والـ POS
doc = nlp(cleaned_text)

print(f"{'الكلمة':<15} | {'الـ Lemma':<15} | {'الـ POS Tag':<10}")
print("-" * 45)

for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text:<15} | {word.lemma:<15} | {word.upos:<10}")