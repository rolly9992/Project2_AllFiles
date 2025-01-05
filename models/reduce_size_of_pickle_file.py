# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 09:35:24 2025

@author: Milo_Sand
"""

import pickle
import gzip
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize(text):
    '''
    

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.

    Returns
    -------
    clean_tokens : TYPE
        DESCRIPTION.

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load the machine learning model
with open(r'C:\Users\Milo_Sand\Documents\GitHub\Project2\models\trainedmodel.pkl', 'rb') as f:
    model = pickle.load(f)

#Reduce the precision of numerical values
#model.weights = np.around(model.weights, decimals=4)

#Remove unnecessary data
#del model.dataset

# Compress the pickle file
#with gzip.open('trainedmodel.pkl.gz', 'wb') as f:
#    pickle.dump(model, f)
import json 
json_model = json.dumps(model.__dict__)
# Save the JSON data to a file
with open('testmodel.json', 'w') as f:
    f.write(json_model)

with open('testmodel.json', 'r') as f:
    json_model = json.load(f)