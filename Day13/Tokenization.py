import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "AI is transforming the world!"
tokens = word_tokenize(text)
print(tokens)
