from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "AI is transforming the world!"
tokens = tokenizer.tokenize(text)
print(tokens)