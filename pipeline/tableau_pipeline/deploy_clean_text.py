# Import packages
from tabpy.tabpy_tools.client import Client

client = Client("http://10.155.94.140:9004/")

def clean_text(text_list):
    """
    Clean text with spacy library
    """
    import pandas as pd
    import numpy as np
    import spacy

    # Import language model
    nlp = spacy.load('en_core_web_sm', disable=['tagger','parser', 'ner'])

    #Load custom stopwords
    custom_stopwords = []
    with open("/home/nusintern/project/nus/scripts/stopwords.txt") as f:
        custom_stopwords = f.read().splitlines()
    custom_stopwords = set(custom_stopwords)
    return [process_text(x, nlp, custom_stopwords) for x in text_list]

def process_text(text, nlp, stopwords):
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
    if not text:
        return ""
    doc = nlp(text)
    filtered = [token.lemma_ for token in doc if (token.is_stop == False and token.is_alpha and token.is_ascii and  token.like_url == False and token.like_email == False)] # remove stopwords, non-alpha tokens
    tokens = [w.lower() for w in filtered] #lower case
    tokens = [w for w in tokens if w not in stopwords] # remove custom stopwords
    processed_text = ' '.join(tokens) #detokenized
    return processed_text    

client.deploy("clean_text", clean_text, 'Returns processed text using Spacy library', override = True)