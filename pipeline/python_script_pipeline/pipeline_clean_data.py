# Import packages
import pandas as pd
import numpy as np
import os
import json
import re

# Text preprocessing modules
import string
import spacy

# keep only relevant columns to reduce data size 
def column_selection(df):
    columns = ['CLOSE_DT',
        'OPEN_DT',
        'AREA',
        'PRIO_CD',
        'RESOLUTION_CD',
        'SEV_CD',
        'SUBTYPE_CD',
        'SUB_AREA',
        'TYPE_CD',
        'W_AREA_CODE',
        'X_PROD_VERSION',
        'X_PRODUCT',
        'X_ENTL_TYPE',
        'X_SR_TITLE',
        'X_SLM_DUE_DT',
        'X_ENTL_MTRC_UNIT',
        'X_ENTL_MTRC_VALUE',
        'X_FIRST_RESPONSE_DT',
        'X_SR_PRODUCT_FAMILY',
        'X_PAR_COMMENTS',
        'PROCESSED_PAR',
        'SR_NUM']
    df = df[columns]


def text_processing(df):

    nlp = spacy.load('en_core_web_sm', disable=['tagger','parser', 'ner'])

    custom_stopwords = []
    with open("stopwords.txt") as f:
        custom_stopwords = f.read().splitlines()
    custom_stopwords = set(custom_stopwords)

    df['Processed_Abstract'] = df['Abstract'].apply(lambda x: process_text_spacy(x, nlp, custom_stopwords))
    df['Processed_PAR'] = df['Problem, Analysis, Resolution'].apply(lambda x: process_text_spacy(str(x), nlp, custom_stopwords))    


def process_text_spacy(text, nlp, custom_stopwords):
    '''
    This function performs text data preprocessing, including tokenizing the text, converting text to lower case, removing
    punctuation, removing digits, removing stop words, stemming the tokens, then converting the tokens back to strings.
    
    Args:
    ------
        text (string): the text data to be processed
    
    Returns:
    --------
        Returns processed text (string)
    '''
     doc = nlp(text)
    filtered = [token.lemma_ for token in doc if (token.is_stop == False and token.is_alpha and token.is_ascii and  token.like_url == False and token.like_email == False)] # remove stopwords, non-alpha tokens
    tokens = [w.lower() for w in filtered] #lower case
    tokens = [w for w in tokens if w not in custom_stopwords] # remove custom stopwords
    processed_text = ' '.join(tokens) #detokenized
    return processed_text

