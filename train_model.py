import nltk
import json
import numpy as np
import tensorflow as tf
import random
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Initialize lemmatizer
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

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

X_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(y_train[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save trained model
model.save("health_assistant_model.h5")

print("✅ Model trained and saved as 'health_assistant_model.h5' successfully!")
