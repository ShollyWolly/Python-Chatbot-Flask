import random
import json
import pickle
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import googlesearch

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict([bow])[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(return_list)
    return return_list
    

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == "google":
            print("Sholly: Do you want me to Google something for you?")
            GoogleYesNo = input("")
            GoogleYesNo = GoogleYesNo.lower()
            if GoogleYesNo == "yes" or GoogleYesNo == "ye" or GoogleYesNo == "y":
                print("Sholly: What exactly do you want me to google for you?")
                message = input("")
                Search_list = []
                for x in googlesearch.search(message, tld="com", lang='en', num=5, stop=5, pause=2):
                    Search_list.append(x)
                tmp = ""
                for y in Search_list:
                    tmp = tmp + y + "\n"
                tmp = tmp + "Is there anything else I can do for you?"
        
                result = random.choice(i["responses"])
                result = "Sholly: " + result + "\n" + tmp
                break
            elif GoogleYesNo == "no" or GoogleYesNo == "n":
                result = "Sholly: Sorry my bad, if there is anything else I can help you with, let me know."
                break
            else:
                result = "Sorry I didn't understand that"
                break

        elif i["tag"] == tag:
            result = "Sholly: " + random.choice(i["responses"])
            break
    return result

print("Sholly is now at your service!")

while True:
    message = input("You: ")
    message = message.lower()
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)