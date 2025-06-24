from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

items = [
  "Action movie with explosions",
  "Romantic comedy with happy ending",
  "Action thriller with spies and guns"

]


vectorizer= TfidfVectorizer()
item_vectors =vectorizer.fit_transform(items)

#Calcu;ate similarity between first item and others.
similarity= cosine_similarity(item_vectors[0:1],item_vectors).flatten()
print("\nCosine Similarity Matrix:\n", similarity)

#Recommend the most similar items(excluding itself)
recommendations = np.argsort(-similarity)[1:]
print(f"items similar to '{items[0]}':")
for idx in recommendations:
  print(f"-{items[idx]} (score: {similarity[idx]:.2f})")

