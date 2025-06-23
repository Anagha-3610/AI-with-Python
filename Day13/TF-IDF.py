from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "AI is amazing",
    "Machine learning is a part of AI",
    "AI and machine learning are related"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
