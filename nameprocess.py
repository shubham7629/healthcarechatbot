from flask import jsonify
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model4.h5')
import json
import random
intents = json.loads(open('dataset2.json', encoding='utf-8').read())
words = pickle.load(open('textdisease1.pkl','rb'))
classes = pickle.load(open('labelsdisease1.pkl','rb'))

ignore_words = ['?','!','_','-','give','me','tell','about','what','is','know']


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_words]
                        
    return sentence_words


def bow(sentence, words, show_details=True):
    
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    return_list = []
    for r in results:
        return_list.append({"Disease_Num": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
   
    list_of_intents = intents_json['intents']
    result =[]
    for intent in ints:

        for i in list_of_intents:
            if(i['Disease_Num']== intent["Disease_Num"]):
                result.append({"Disease": i["Disease"][0],"Symptoms": i["Symptoms"]})
            
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


final_res = chatbot_response("Symptoms of paralysis")
print(final_res)






