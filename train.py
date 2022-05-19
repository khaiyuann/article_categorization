# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:12:52 2022

This script is used to train a NLP deep learning model that categorizes
articles into 5 categories based on its topic.

@author: LeongKY
"""
#%% Import s
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from categorization_modules import CategorizationModeller

#%% statics
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model', 'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(), 'saved_model', 'tokenizer.json')
ENC_PATH = os.path.join(os.getcwd(), 'saved_model', 'encoder.pkl')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

#%% 1. Load data
df = pd.read_csv(URL)

#separate feature (text) and label (category)
text = df['text']
cat = df['category']

#%% 2. Inspect data
print(df.head(10))
print(cat.value_counts())
#observed that dataset is fairly balanced between 386 and 511 entries per cat

#%% 3. Clean data
# obtain list containing length of each article before splitting
len_list = [len(s.split()) for s in text]

mod = CategorizationModeller()
text = mod.filter_text(text) # split and filter only lower lowercase letters
    
#%% 4. Preprocess data
num_words = 10000
oov_token = '<OOV>'

# tokenize vectorize words
token = mod.tokenize_text(text, num_words, oov_token, TOKENIZER_PATH)

# vectorize sequence of texts
avg_len = int(sum(len_list)/len(len_list)) #avg length of articles
text = mod.vectorize_text(text, avg_len, token)

# encode label with OHE
cat = mod.ohe_target(cat, ENC_PATH)

# split train test
X_train, X_test, y_train, y_test = train_test_split(text, cat, 
                                                    test_size=0.3, 
                                                    random_state=19)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

#%% 5. Create model
model = mod.create_model(nb_embed=64, nb_lstm=32, dropout=0.3, 
                         nb_classes=5, num_words=num_words)

# callbacks
log_files = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))

es_callback = EarlyStopping(monitor='val_loss', patience=3)
tb_callback = TensorBoard(log_dir=log_files)
callbacks = [es_callback, tb_callback]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# train model
hist = model.fit(X_train, y_train, epochs=50,
                 validation_data=(X_test, y_test),
                 callbacks=callbacks)

#%% 7. Evaluate model
mod.evaluate_model(model, X_test, y_test, nb_classes=5)

# save model for deployment
model.save(MODEL_SAVE_PATH)