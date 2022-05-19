# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:41:57 2022

This class file contains the functions defined for use in training and
deployment of a NLP deep learning article categorization model.

@author: LeongKY
"""
import os
import re
import json
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report,\
    accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding,\
    Bidirectional
from tensorflow.keras.utils import plot_model

class CategorizationModeller():
    def __init__(self):
        pass
    
    def filter_text(self, text):
        '''
        This function is used to split and filter provided text to obtain only 
        letters in lowercase.

        Parameters
        ----------
        text : list
            list containing text to split and filter.

        Returns
        -------
        text : list
            list containing split and filtered text.

        '''
        for index, txt in enumerate(text):
            # split and filter to only lowercase text
            text[index] = re.sub('[^a-zA-Z]', ' ', txt).lower().split()
            
        return text
    
    def tokenize_text(self, text, num_words, oov_token, path):
        '''
        This function is used to tokenize provided text and export the
        resulting tokenizer in .json format.

        Parameters
        ----------
        text : list
            list containing split text.
        num_words : int
            dictionary size of tokenizer.
        oov_token : str
            token for out of vocabulary text.
        path : path
            directory to export tokenizer in .json format.

        Returns
        -------
        token : Tokenizer object
            Tokenizer object fitted onto provided text.

        '''
        token = Tokenizer(num_words=num_words, oov_token=oov_token)
        token.fit_on_texts(text)

        # save tokenizer data to json for deployment
        token_json = token.to_json()
        with open(path, 'w') as json_file:
            json.dump(token_json, json_file)
            
        return token
    
    def vectorize_text(self, text, maxlen, token):
        '''
        This function vectorizes the provided text using the token and then
        pads it to a uniform length.

        Parameters
        ----------
        text : list
            list containing split text.
        maxlen : int
            maximum length of each item in list.
        token : Tokenizer object
            Tokenizer to vectorize text.

        Returns
        -------
        text : array
            array containing vectorized and padded text.

        '''
        text = token.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=maxlen, 
                             padding='post', truncating='post')
        
        return text
    
    def ohe_target(self, target, path):
        '''
        This function is used to encode the target using One Hot encoding and
        saves the fitted encoder into .pkl format.

        Parameters
        ----------
        target : str
            target data.
        path : path
            directory to save the encoder in .pkl format.

        Returns
        -------
        target : array
            array containing encoded target.

        '''
        ohe = OneHotEncoder(sparse=False)
        target = ohe.fit_transform(np.expand_dims(target, -1))
        pickle.dump(ohe, open(path, 'wb'))
        
        return target
    
    def create_model(self, nb_embed, nb_lstm, dropout, nb_classes, num_words):
        '''
        This function is used to create a NLP model with an embedding layer 
        and 2 bidirectional LSTM layers for categorizing article topics.

        Parameters
        ----------
        nb_embed : int
            embedding layer parameter.
        nb_lstm : int
            LSTM layer parameter.
        dropout : float
            dropout parameter.
        nb_classes : int
            number of target classes.
        num_words : int
            number of words in tokenizer dictionary.

        Returns
        -------
        model : model
            NLP model created based on provided parameters.

        '''
        model = Sequential()
        model.add(Embedding(num_words, nb_embed))
        model.add(Bidirectional(LSTM(nb_lstm, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nb_lstm)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='softmax'))
        plot_model(model, os.path.join(os.getcwd(), 'results', 'model.png'))
        model.summary()
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, nb_classes):
        '''
        This function is used to evaluate the performance of the model and
        provides the confusion matrix, f1 score and accuracy.

        Parameters
        ----------
        model : model
            model to be evaluated.
        X_test : array
            feature test data.
        y_test : array
            target test data.
        nb_classes : int
            number of target classes.

        Returns
        -------
        None.

        '''
        # predict using model
        y_pred_adv = np.empty([len(X_test), nb_classes])
        for index, test in enumerate(X_test):
           y_pred_adv[index,:] = model.predict(np.expand_dims(test, 0))
        
        # model scoring
        y_pred_res = np.argmax(y_pred_adv, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # model evaluation
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_true, y_pred_res))
        print('\nClassification Report:\n')
        print(classification_report(y_true, y_pred_res))
        print('\nThis model has an accuracy of ' 
              + str('{:.2f}'.format(accuracy_score(y_true, y_pred_res)*100))
              +' percent')