import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time

start_time = time.time()

# read data
print("Reading df...")

df = pd.read_csv('../combine_sr.csv',  usecols=['F5_Product','Processed_PAR','Processed_Abstract'])
df.dropna(subset = ['Processed_PAR'], inplace = True)

# Processed PAR column
sentences = df.Processed_PAR.to_list()

# Create sentence embeddings for each reviews.
print('Create embeddings ...')
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
sentence_embeddings = model.encode(sentences)

# save sentence embeddings
print("save embeddings ...")
with open('cluster_embedding.npy', 'wb') as f:
    np.save(f, sentence_embeddings)

end_time = time.time()
print("Total Time used: {} minutes".format((end_time - start_time)//60))
