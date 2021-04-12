def clustering(text_list, product_type):
    """
    Cluster PAR column for each product and return their cluster label
    """
    import pandas as pd
    import numpy as np
    import hdbscan
    import umap
    from sentence_transformers import SentenceTransformer
    
    df = pd.DataFrame(list(zip(text_list, product_type)), columns = ['PAR', 'Product'])
    df.dropna(subset = ['PAR'], inplace = True)

    # set default cluster label to -1
    df['cluster_label'] = -1
    
    # get list of products
    products = df.Product.value_counts().index
    
    # model for embedding
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    print("enumerating to cluster data")
    for i, product in enumerate(products):
        print("starting to cluster product: " + str(product))
        filtered_df = df[df.Product == product]
        
        # must be smaller than n_component to prevent error in umap
        if len(filtered_df) < 64:
            continue
        
        # encode text with embedding
        filtered_embedding = model.encode(filtered_df.PAR.tolist())
        
        print("reduce dimension")
        # dimenstinality reduction
        reduce_embedding = umap.UMAP(n_components = 64).fit_transform(filtered_embedding)
        
        # clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size = 70, min_samples = 1, cluster_selection_epsilon = 1.3)
        cluster_labels = clusterer.fit_predict(reduce_embedding)
        df.loc[df.Product == product, 'cluster_label'] = cluster_labels
    return df['cluster_label'].to_list()