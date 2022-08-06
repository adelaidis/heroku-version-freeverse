import numpy as np
import random
import re
from profanity_check import predict
import os
import joblib
from config.definitions import ROOT_DIR


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf 

import requests

from keras.models import load_model


tokenizer = Tokenizer()
  
def load_keras_model():
    global model
    #model = tf.keras.models.load_model('C:/Users/magie/OneDrive/Documents/Studia/TM470/TM470project/prototype1/models/25-07/LSTM256_voc19709_seq50_emb64_5.h5')
    #model = tf.keras.models.load_model('C:/Users/magie/OneDrive/Documents/Studia/TM470/TM470project/prototype1/models/25-07/mymodel51-75.h5')
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'static', 'mymodel51-75.h5'))
    # Graph is needed to execute tensor computations: 
    # https://www.tensorflow.org/guide/intro_to_graphs
    global graph
    # Need to use compat.v1 versioning to fix AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
    graph = tf.compat.v1.get_default_graph()
    return model

def prepare_tokenization():
    VOCAB_SIZE = 25000
    dataset = open(os.path.join(ROOT_DIR, 'static', 'lastEDITEDfixed.txt'), encoding='utf-8').read()
    #tokenizer = Tokenizer(num_words=VOCAB_SIZE, lower=True)
    #tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') - probably not needed
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    train_poetry = dataset.splitlines()
    tokenizer.fit_on_texts(train_poetry)
    return tokenizer

def create_poem(model, tokenizer, seed):

    next_words = random.randrange(10, 250)
    # Remove most special signs from the seed
    poem = re.sub(r"[^a-zA-Z0-9' ]","", seed.lower())
    previous_word = ""

    for word in range(next_words):
        encoded_seed = tokenizer.texts_to_sequences([poem])
        encoded_seed = pad_sequences(encoded_seed, maxlen=50, padding='pre')
        
        # Use argmax to find the most probable class
        y_pred = np.argmax(model.predict(encoded_seed), axis=-1)

        predicted_word = ""
        # find word in dictionary
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                if word != previous_word:
                    previous_word = word
                    if word[-3:] != "eov":
                        predicted_word = word
                    else:
                        predicted_word = word.replace("eov","\n")
                else:
                    try:
                       # previous_word = word = predicted_word = get_synonym(word.replace("eov","")).lower()
                       # Get and clean synonym
                        previous_word = word = predicted_word = re.sub(r"[^a-zA-Z0-9' ]","", get_synonym(word.replace("eov","")).lower())
                    except Exception:
                        try:
                            #previous_word = word = predicted_word = get_random_word().lower()
                            # Get and clean random word
                            previous_word = word = predicted_word = re.sub(r"[^a-zA-Z0-9' ]","", get_random_word().lower())
                        except Exception:
                            predicted_word = "\n"      
        # add predicted word to seed
        poem = poem + ' ' + predicted_word
       
    print("poem = " + poem)
    return poem

def get_synonym(word):
    response = requests.get("https://words.bighugelabs.com/api/2/2da4fa3679f36cded7f904c1b08b614c/" + word + "/json")
    data = response.json()
    if "noun" in data:
        if "syn" in data["noun"]:
            # Take random noun from json data
            synonym = data["noun"]["syn"][random.randrange(0, len(data["noun"]["syn"])-1)]
    elif "verb" in data:
        if "syn" in data["verb"]:
            # Take random verb from json data
            synonym = data["verb"]["syn"][random.randrange(0, len(data["verb"]["syn"])-1)]
    elif "verb" and "noun" and "syn" not in data:
        # Take random word from json data
        synonym = data[random.randrange(0, len(data)-1)]
    synonym = str(synonym)
    return check_offensiveness(synonym)

def get_random_word():
    response = requests.get("https://random-word-api.herokuapp.com/word")
    data = response.json()
    random_word = str(data[0])
    return check_offensiveness(random_word)

def check_offensiveness(word):
    if predict([word]) == 0:
        return word
    else:
        get_random_word()


#TODO: save api keys separately:
# https://stackoverflow.com/questions/56995350/best-practices-python-where-to-store-api-keys-tokens

     
 

