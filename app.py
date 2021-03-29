# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:04:32 2021

@author: Dorra
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle


from tensorflow.keras.models import load_model

import tensorflow as tf
import re
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Dense, Activation, Flatten
#from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Conv1D, MaxPooling1D,SpatialDropout1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional,LSTM,Dense, Activation, Flatten
#from keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#return (render_template('result.html',prediction = 1))
        model = load_model('weights.hdf5')
	return (render_template('result.html',prediction = 1))
##loading
        #with open('tokenizer.pickle', 'rb') as handle:
		#tokenizer = pickle.load(handle)
#"convert tokens to indices
        #def tokenize_tweets(text):
		#return tokenizer.convert_tokens_to_ids(['[CLS]'] + text + ['[SEP]'])
        #if (request.method == 'POST'):
		#message = request.form['message']
		#data = message
                ## supprimer les caractères répétitifs dans un mot
                #tweet=re.sub(r'(.)\1+', r'\1', data)
                ## convert words to tokens
                #tokenized_tweet= tokenizer.tokenize(tweet)
                #encoded_tweet = [tokenize_tweets(tokenized_tweet)]
                #padded_tweet = pad_sequences(encoded_tweet, maxlen=4000, padding='post', truncating='post')
                ##make prediction
                #my_prediction= model.predict(padded_tweet)
                #pred_final = np.argmax(my_prediction,axis=1)
                #return (render_template('result.html',prediction = pred_final))



if __name__ == '__main__':
    app.run(debug=True)
