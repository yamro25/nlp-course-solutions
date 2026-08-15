from collections import Counter

# 1. نص للتجربة
text = "nlp is amazing and nlp is essential for artificial intelligence"

# 2. تقسيم النص إلى كلمات
words = text.lower().split()

# 3. حساب تكرار الكلمات
word_counts = Counter(words)

# 4. طباعة النتيجة
print("--- Words Frequency ---")
for word, count in word_counts.items():
    print(f"{word}: {count}")