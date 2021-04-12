from gsdmm import MovieGroupProcess
from utils import compute_V, evaluate
import umap
import numpy as np
import pandas as pd

np.random.seed(123)
# read data
df = pd.read_csv('../combine_sr.csv', usecols=['F5_Product', 'Processed_PAR'])
df = df.dropna(subset=['Processed_PAR'], inplace=True)

# load embedding
with open('cluster_embedding.npy', 'rb') as f:
    embedding = np.load(f)
df['embedding'] = embedding.tolist()

# get number of records per product
num_records_product = df.groupby('F5_Product').count().sort_values(by='Processed_PAR', ascending=False)
# save the top 10 product names 
top_ten_product = num_records_product.head(10).index.tolist()
print(top_ten_product)

for i in range(len(top_ten_product)):
    print('{}/10 processing {}'.format(i+1, top_ten_product[i]))
    df_product = df[df.F5_Product == top_ten_product[i]].copy()
    product_embedding = np.array(df_product.embedding.to_list())

    # dimensionality reduction
    print('dimension reduction')
    reduced_embedding = umap.UMAP(n_components = 32).fit_transform(product_embedding)

    # split document (each PAR) into list of unique tokens as required for gsdmm model input
    texts = [list(set(text.split())) for text in df_product.Processed_PAR]
    # compute number of unique words in the vocabulary
    V = compute_V(texts)

    # build GSDMM model
    ## n_iters=10, number of clusters drops quickly and gets stable within 10 iterations
    ## K=50, K must be greater than number of ground truth clusters. We assume there are at most 50 topics for each product
    ## alpha=range(0,1), performance is stable within the range, but when alpha = 0, GSDMM converges very quickly
    ## beta=range(0,0.2), number of clusters drop as beta increases, performance is stable within the range
    mgp = MovieGroupProcess(K=50, n_iters=10, alpha=0.1, beta=0.1)
    # fit model and return list of labels for each document
    cluster_labels = mgp.fit(texts, V)
    # get silhouette_score
    s_score = evaluate(reduced_embedding, cluster_labels)
    print('{} processing completed, s_score={}'.format(top_ten_product[i], s_score))

    # append cluster label to dataframe
    df_product['cluster_labels'] = cluster_labels
    df_product.to_csv('./data/{}_labels.csv'.format(top_ten_product[i]))
