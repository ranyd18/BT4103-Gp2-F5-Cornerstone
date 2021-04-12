# Import packages
from tabpy.tabpy_tools.client import Client
import pandas as pd
import numpy as np
import hdbscan
import umap
from sentence_transformers import SentenceTransformer

def get_cluster_label(text_list, product_type):
    """
    Cluster PAR column for each product and return their cluster label
    """
    
    df = pd.DataFrame(list(zip(text_list, product_type)), columns = ['PAR', 'Product'])
    # df.dropna(subset = ['PAR'], inplace = True)

    # set default cluster label to -2
    df['cluster_label'] = -2
    
    # get list of products
    products = df.Product.value_counts().index
   
    # model for embedding
    print("creating embedding")
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    print("enumerating to cluster data")
    for i, product in enumerate(products):
        print("starting to cluster product: " + str(product))
        filtered_df = df[(df.Product == product) & (df["PAR"].notnull())]
        
        
        # encode text with embedding
        filtered_embedding = model.encode(filtered_df.PAR.tolist())
        
        # dimenstinality reduction
        print("reduce dimension")
        reduce_embedding = filtered_embedding
        
        try:
             # dimenstinality reduction
            reduce_embedding = umap.UMAP(n_components = 64).fit_transform(filtered_embedding)
             # clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size = 40, min_samples = 1)
            cluster_labels = clusterer.fit_predict(reduce_embedding)
            # update dataframe
            df.loc[(df.Product == product) & (df["PAR"].notnull()), 'cluster_label'] = cluster_labels
        except:
            print("Exception occured when clustering")
        
    return df['cluster_label'].to_list()


def clustering(input_df):
    '''
    This function create the columns with cluster number and output the new dataframe.
    Called in Tableau Prep
    Args:
    ------
        whole dataframe from Tableau
    
    Returns:
    --------
        Returns processed pandas dataframe
    '''
    cluster = get_cluster_label(input_df['PROCESSED_PAR'].tolist(), input_df['X_PRODUCT'].tolist())
    input_df['CLUSTER'] = cluster
    output_df = input_df
    # return the entire df
    return output_df    

def get_output_schema():
    return pd.DataFrame({
        'CLOSE_DT': prep_datetime(),
        'OPEN_DT': prep_datetime(),
        'AREA': prep_string(),
        'PRIO_CD': prep_string(),
        'RESOLUTION_CD': prep_string(),
        'SEV_CD': prep_string(),
        'SUBTYPE_CD': prep_string(),
        'SUB_AREA': prep_string(),
        'TYPE_CD': prep_string(),
        'W_AREA_CODE': prep_string(),
        'X_PROD_VERSION': prep_string(),
        'X_PRODUCT': prep_string(),
        'X_ENTL_TYPE': prep_string(),
        'X_SR_TITLE': prep_string(),
        'X_SLM_DUE_DT': prep_datetime(),
        'X_ENTL_MTRC_UNIT': prep_string(),
        'X_ENTL_MTRC_VALUE': prep_string(),
        'X_FIRST_RESPONSE_DT': prep_string(),
        'X_SR_PRODUCT_FAMILY': prep_string(),
        'X_PAR_COMMENTS': prep_string(),
        'PROCESSED_PAR': prep_string(),
        'SR_NUM': prep_string(),
        'CLUSTER': prep_int()
    })