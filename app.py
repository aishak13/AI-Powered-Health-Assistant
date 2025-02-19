from flask import Flask, request, jsonify
import nltk
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load trained model
model = tf.keras.models.load_model("health_assistant_model.h5")

# Load words and classes
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intent_list, intents_json):
    tag = intent_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand. Please consult a doctor."

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chatbot_response():
    user_input = request.json["message"]
    predicted_intent = predict_class(user_input)
    response = get_response(predicted_intent, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
