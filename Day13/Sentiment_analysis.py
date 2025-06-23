from textblob import TextBlob

text = "I love AI! It's incredibly powerful."
blob = TextBlob(text)

print(blob.sentiment)
