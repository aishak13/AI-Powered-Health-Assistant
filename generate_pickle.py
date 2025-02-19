import nltk
import json
import pickle
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load dataset
with open("intents.json", "r") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("words.pkl and classes.pkl generated successfully!")
