import nltk
nltk.download('popular')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import googlesearch
from bs4 import BeautifulSoup
import requests
import re

from keras.models import load_model

import json
import random

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
WordList = ["google", "look up", "search for"]

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json, msg):
    tag = ints[0]['intent']
    # check if it's a google command, add temp word, so it can't be out of index
    msg = msg + " tmp"
    if msg.split()[0] == "google" or msg.split()[0] + " " + msg.split()[1] == "look up" or msg.split()[0] + " " + msg.split()[1] == "search for":
        tag = "google"
    msg = msg.replace(" tmp","",1)
    # print(msg)

    list_of_intents = intents_json['intents']

    # search for response
    for i in list_of_intents:

        #if response google, do wikipedia check, do google seach and format text with regexp
        if i["tag"] == "google":
            for index in WordList:
                if index in msg:
                    msg = msg.replace(index,"",1)

            Search_list = []
            for x in googlesearch.search(msg, tld="com", lang='en', num=3, stop=3, pause=2):
                Search_list.append(x)
            # print(Search_list)
            
            index = 0
            for zahl in Search_list:
                print(zahl)
                if "wikipedia.org" in zahl:
                    url = zahl
                    Page = requests.get(url)
                    doc = BeautifulSoup(Page.text, "html.parser")
                    tmp = doc.find_all("p")
                    for another_index in tmp:
                        if another_index.text != "\n":
                            tmp = re.sub('\[(.*?)\]|\((.*?)\)|\{(.*?)\}','',another_index.text)
                            break

                    break
                else:
                    tmp = ""
                    for y in Search_list:
                        tmp =  tmp + "<a>" + y + "</a>" + "\n"
                    
                index = index + 1
            
            result = random.choice(i["responses"])
            result = result + "\n" + tmp + "Is there anything else I can do for you?"
            print(result)
            break

        elif (i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    msg = msg.lower()

        # res = msg
    ints = predict_class(msg, model)
    res = getResponse(ints, intents, msg)
    return res