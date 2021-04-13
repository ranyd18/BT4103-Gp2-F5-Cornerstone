import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import hdbscan
import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", help = "version number")


def cluster_distribution(df, product, column, version = 1):
    # group by cluster label
    cluster = df.groupby('cluster_label').count()
    # visualize distribution of clusters
    cluster['cluster_label'] = cluster.index
    plt.bar(cluster.cluster_label, cluster[column])
    plt.title('Number of records per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()
    # save figure
    plt.savefig('img/{}_cluster_distribution_v{}.png'.format(product, version))
    plt.close()
    return cluster

def world_cloud_plot(df, cluster, product, column, version = 1):
    top_nine = cluster.sort_values(by=column, ascending=False).head(9)
    cluster_num = top_nine.index.tolist()
    stopwords = ['big', 'ip', 'fail', 'device', 'issue', 'error', 'vip', 'need', 'assistance', 'resolve']
    
    axes=[]
    fig=plt.figure(figsize = (20,10))

    for i in range(min(9, len(cluster_num))):
        num = cluster_num[i]
        df_filter = df[df.cluster_label == num]
        text = " ".join(str(item) for item in df_filter[column])
        wordcloud = WordCloud(background_color="white", stopwords=stopwords, max_words=50).generate(text)

        axes.append(fig.add_subplot(3, 3, i+1) )
        plt.imshow(wordcloud, interpolation='nearest')

    fig.tight_layout()    
    plt.show()
    plt.savefig('img/{}_wordcloud_v{}.png'.format(product, version))
    plt.close()


if __name__ == '__main__':
    
    # read data
    df = pd.read_csv('../combine_sr.csv', usecols = ['F5_Product', 'Processed_PAR'])
    df.dropna(subset = ['Processed_PAR'], inplace = True)

    # get version
    args = parser.parse_args()
    version = args.version

    # load embedding
    with open('cluster_embedding.npy', 'rb') as f:
        embedding = np.load(f)

    df['embedding'] = embedding.tolist()

    print("product segmentation")
    print(df.F5_Product.value_counts()[:10])

    products = df.F5_Product.value_counts().index[:10]
    
    for i, product in enumerate(products):
        print("Processing product {}".format(i+1))
        start_time = time.time()
        filtered_df = df[df.F5_Product == product].copy()
        filtered_embedding = np.array(filtered_df.embedding.to_list())

        # dimensionality reduction
        print("dimension reduction")
        reduce_embedding = umap.UMAP(n_components = 64).fit_transform(filtered_embedding)

        # clustering
        print("clustering")
        clusterer = hdbscan.HDBSCAN(min_cluster_size = 40, min_samples = 1)
        cluster_labels = clusterer.fit_predict(reduce_embedding)
        filtered_df['cluster_label'] = cluster_labels
        
        
        # visualisation
        column = 'Processed_PAR'
        cluster = cluster_distribution(filtered_df, product, column, version)
        world_cloud_plot(filtered_df, cluster, product, column, version)

        print("--- clustering runs for  %s mins -----" % ((time.time() - start_time)/60))
