import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import json
import random
import re
import joblib
from profanity_check import predict, predict_prob
import os
from config.definitions import ROOT_DIR


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf 

import requests

from keras.models import load_model


tokenizer = Tokenizer()
  
def load_keras_model():
    global model
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'static', 'mymodel76-100.h5'))
    # Graph is needed to execute tensor computations: 
    # https://www.tensorflow.org/guide/intro_to_graphs
    global graph
    # Need to use compat.v1 versioning to fix AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
    graph = tf.compat.v1.get_default_graph()
    return model

def prepare_tokenization():
    VOCAB_SIZE = 18261
    dataset = open(os.path.join(ROOT_DIR, 'static', 'allCorpusLinesLAST.txt'), encoding='utf-8').read()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    train_poetry = dataset.splitlines()
    tokenizer.fit_on_texts(train_poetry)
    return tokenizer

def create_poem(model, tokenizer, seed):
    random_count = 0
    synonym_count = 0
    next_words = 150
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
                    if synonym_count < random.randrange(0,3):
                        try:
                        # Get and clean synonym
                            previous_word = word
                            synonym = re.sub(r"[^a-zA-Z0-9' ]","", get_thesaurus_synonym(word.replace("eov","")).lower())
                            if synonym not in seed:
                                predicted_word = word = synonym
                                synonym_count += 1
                            else: 
                                previous_word, word, predicted_word, random_count = process_random_word(previous_word, word, predicted_word, random_count, seed)  
                        except Exception:
                            if random_count < random.randrange(0,3):
                                previous_word, word, predicted_word, random_count = process_random_word(previous_word, word, predicted_word, random_count, seed)
                    else:
                        if random_count < random.randrange(0,2):
                                    previous_word, word, predicted_word, random_count = process_random_word(previous_word, word, predicted_word, random_count, seed)
                        else:
                            break
                            

        # add predicted word to seed
        poem = poem + ' ' + predicted_word
       
    print("poem = " + poem)
    return check_poem_offensiveness(poem)

def get_thesaurus_synonym(word):
    response = requests.get("https://words.bighugelabs.com/api/2/2da4fa3679f36cded7f904c1b08b614c/" + word + "/json")
    data = response.json()
    if "adjective" in data:
        if "syn" in data["adjective"]:
            # Take random adjective from json data
            synonym = data["adjective"]["syn"][random.randrange(0, len(data["adjective"]["syn"]))]
    elif "noun" in data:
        if "syn" in data["noun"]:
            # Take random noun from json data
            synonym = data["noun"]["syn"][random.randrange(0, len(data["noun"]["syn"]))]
    elif "verb" in data:
        if "syn" in data["verb"]:
            # Take random verb from json data
            synonym = data["verb"]["syn"][random.randrange(0, len(data["verb"]["syn"]))]
    elif "verb" and "noun" and "syn" not in data:
        # Take random word from json data
        synonym = data[random.randrange(0, len(data))]
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
        return get_random_word()

def check_poem_offensiveness(poem):
    if predict_prob([poem]) < 0.5:
        return poem
    else:
        return "Roses are red \n violets are blue \n this poem wasn't nice \n so try something new"

def process_random_word(previous_word, word, predicted_word, random_count, seed):
    try:
        previous_word = word
        # Get and clean random word
        word = predicted_word = re.sub(r"[^a-zA-Z0-9' ]","", get_random_word().lower())
        random_count += 1
    except Exception:
        try: 
            randomSeedWordIndex = int(random.randrange(0, len(seed.split())))
            # Get and clean synonym of random word of the seed
            predicted_word = word = re.sub(r"[^a-zA-Z0-9' ]","", get_thesaurus_synonym(seed.split()[randomSeedWordIndex]).lower())
            previous_word = word
        except Exception:
            predicted_word = ""
    return previous_word, word, predicted_word, random_count
