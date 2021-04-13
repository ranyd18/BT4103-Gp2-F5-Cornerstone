import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import hdbscan
import umap
from sklearn.metrics import silhouette_score


def eval(X, labels):
    # measured in silhouette index
    # return the mean Silhoutte coefficient over all samples
    # the best value is 1 and the worst value is -1
    # values near 0 indictae overlapping clusters
    # negative values generally indicate a sample has been assigned to the wrong cluster, as a different cluster is more similar
    
    # X:= cluster objects as numpy array
    # labels := cluster labels as numpy array
    return silhouette_score(X, labels, metric = 'sqeuclidean')

"""
Parameter selection for the clustering algo:
https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size
- min_cluster_size: minimum size of a resulting cluster
- min_samples: the larger the values of min_samples, the more conservative the coustering: more points will be declared as noise and clutsers will be restricted to progressively more dense areas
- cluster_selection_epsilon: set the value of 0.5 if you don't want to separate clusters that are less than 0.5 units apart
- alpha: not recommended to change
"""


df = pd.read_csv('../combine_sr.csv', usecols = ['F5_Product', 'Processed_PAR'])
df.dropna(subset = ['Processed_PAR'], inplace = True)

min_cluster_size = range(10, 60, 20)
min_samples = range(1, 42, 20)
epsilon = [0.1, 0.5, 0.9]

# load embedding
with open('cluster_embedding.npy', 'rb') as f:
    embedding = np.load(f)

df['embedding'] = embedding.tolist()
reduce_embedding = umap.UMAP(n_components = 32).fit_transform(embedding)

# clustering
mcs_ = []
ms_ = []
eps_ = []
sc_ = []
print("Tuning")
for mcs in min_cluster_size:
    for ms in min_samples:
        for e in epsilon:
            mcs_.append(mcs)
            ms_.append(ms)
            eps_.append(e)

            clusterer = hdbscan.HDBSCAN(min_cluster_size = mcs, min_samples = ms, cluster_selection_epsilon = e)
            cluster_labels = clusterer.fit_predict(reduce_embedding)

            # evaluation
            score = eval(reduce_embedding, cluster_labels)
            sc_.append(score)
            print('min_cluster_size: {}, min_samples: {}, epsilon:{}, score: {}'.format(mcs, ms, e, score))

df = pd.DataFrame(zip(mcs_, ms_, eps_, sc_), columns= ['min_cluster_size', 'min_samples', 'epsilon', 'score'])
print(df.head(3))
df.to_csv('tuning_results.csv', index = False)
# Final parameter
# min_cluster_size = 40, min_samples = 1, cluster_selection_epsilon = 1.3