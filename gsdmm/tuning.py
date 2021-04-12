from gsdmm import MovieGroupProcess
from utils import compute_V, evaluate
import umap.umap_ as umap
import numpy as np
import pandas as pd

np.random.seed(123)
# read data
df = pd.read_csv('../product_par.csv') ### CHANGE TO COMBINE_SR.CSV
df = df.dropna(subset=['Processed_PAR'])

# load embedding
print('loading embedding')
with open('cluster_embedding.npy', 'rb') as f:
    embedding = np.load(f)

df['embedding'] = embedding.tolist()
reduced_embedding = umap.UMAP(n_components = 32).fit_transform(embedding)
print('finish loading embedding')

# hyperparameters tuning 
result_dict={}
# split document (each PAR) into list of unique tokens as required for gsdmm model input
texts = [list(set(text.split())) for text in df.Processed_PAR]
# compute number of unique words in the vocabulary
V = compute_V(texts)

# build GSDMM model
## n_iters=10, number of clusters drops quickly and gets stable within 10 iterations
## K=100, K must be greater than number of ground truth clusters. We assume there are at most 100 topics in total
## alpha=range(0,1), performance is stable within the range, but when alpha = 0, GSDMM converges very quickly
## beta=range(0,0.2), number of clusters drop as beta increases, performance is stable within the range
alpha = [a/10 for a in range(1,11)]
beta = [b/20 for b in range(1,5)]
print('start tuning')
for a in alpha:
    for b in beta:
        print('processing a={}, b={}'.format(a, b))
        mgp = MovieGroupProcess(K=100, n_iters=10, alpha=a, beta=b)
        # fit model and return list of labels for each document
        cluster_labels = mgp.fit(texts, V)
        # get silhouette_score
        s_score = evaluate(reduced_embedding, cluster_labels)
        print('s_score={}'.format(s_score))
        result_dict[(a,b)] = s_score

max_key = max(result_dict, key=result_dict.get)
print(result_dict)
print('Best Result: a={}, b={}, s_score={}'.format(max_key[0], max_key[1], result_dict[max_key]))


