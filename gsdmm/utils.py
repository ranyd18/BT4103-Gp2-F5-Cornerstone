from sklearn.metrics import silhouette_score

def compute_V(texts):
    ''' compute number of unique words in the texts '''
    V = set()
    for text in texts:
        for word in text:
            V.add(word)
    return len(V)

def evaluate(X, labels):
    ''' evaluate using silhouette index'''
    return silhouette_score(X, labels, metric='sqeuclidean')