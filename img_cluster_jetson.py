#
# file: img_cluster_jetson.py
# desc: Image Clustering on Jetson Devices
# auth: Mark Swaringen
#
# copyright: Saivvy 2021
#############################################

import pandas as pd
import numpy as np
from PIL import Image

import argparse
import sys
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

from sklearn.metrics import confusion_matrix, rand_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

def find_elbow(df):
    sse = []
    k_list = range(1, 15)
    for k in k_list:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append([k, km.inertia_])
        
    oca_results_scale = pd.DataFrame({'Cluster': range(1,15), 'SSE': sse})
    plt.figure(figsize=(12,6))
    plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
    plt.title('Optimal Number of Clusters using Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

def parse_arguments(args):
    """
    Method that will parse the given arguments and correctly process them
    :param args:
    :return:
    """

    # parse incoming information
    parser = argparse.ArgumentParser(
        description='Test file for testing the RetinaNet. ZOO path is required to load pretrained model.'
    )
    parser.add_argument(
        '--ZOO',
        type=str,
        default=None,
        help='absolute path to the ZOO directory'
    )
    parser.add_argument(
        '--bags',
        type=str,
        default=None,
        help='path to the directory holding the bag files to be checked'
    )

    args = parser.parse_args(args)

    return args

def baseline_test(files_df,count,base_df,scores_df):
    kmeans_base = KMeans(n_clusters=6, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(base_df)
    labels_base = kmeans_base.labels_
    clusters_base = pd.concat([base_df, pd.DataFrame({'cluster_base':labels_base})], axis=1)

    scores_df.loc['Baseline', 'Hopkins'] = hopkins(base_df)
    scores_df.loc['Baseline', 'Silhouette'] = silhouette_score(base_df, kmeans_base.labels_, metric='cosine')
    scores_df.loc['Baseline', 'Data Size'] = sum(clusters_base.memory_usage(deep=True))/1000000

    print('---BASELINE---')
    print('KMeans Silhouette Score: {}'.format(scores_df.loc['Baseline', 'Silhouette']))
    print('Hopkins Score: ',scores_df.loc['Baseline', 'Hopkins'])
    print('Data Size: ',scores_df.loc['Baseline', 'Data Size'])
    acc_score(files_df,count,labels_base,'Baseline')
    return scores_df

def pca_test(files_df,count,base_df,scores_df):
    pca = PCA(n_components=5)
    pca_trans = pca.fit_transform(base_df)
    pca_df = pd.DataFrame(pca_trans)

    kmeans_pca = KMeans(n_clusters=6, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df)
    labels_pca = kmeans_pca.labels_
    clusters_pca = pd.concat([pca_df, pd.DataFrame({'pca_clusters':labels_pca})], axis=1)

    scores_df.loc['PCA', 'Hopkins'] = hopkins(pca_df)
    scores_df.loc['PCA', 'Silhouette'] = silhouette_score(pca_df, kmeans_pca.labels_, metric='cosine')
    scores_df.loc['PCA', 'Data Size'] = sum(clusters_pca.memory_usage(deep=True))/1000000

    print('---PCA---')
    print('KMeans Silhouette Score: {}'.format(scores_df.loc['PCA', 'Silhouette']))
    print('Hopkins Score: ',scores_df.loc['PCA', 'Hopkins'])
    print('Data Size: ',scores_df.loc['PCA', 'Data Size'])
    acc_score(files_df,count,labels_pca,'PCA')
    return scores_df

def tsne_test(files_df,count,base_df,scores_df):
    tsne = TSNE(n_components=3, verbose=1, perplexity=15, n_iter=2000, learning_rate=200,early_exaggeration=4,metric="cosine",init="pca",random_state=42)
    tsne_results = tsne.fit_transform(base_df)
    tsne_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2', 'tsne3'])

    kmeans_tsne = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(tsne_df)
    labels_tsne = kmeans_tsne.labels_
    clusters_tsne = pd.concat([tsne_df, pd.DataFrame({'tsne_clusters':labels_tsne})], axis=1)

    scores_df.loc['t-SNe', 'Hopkins'] = hopkins(tsne_df)
    scores_df.loc['t-SNe', 'Silhouette'] = silhouette_score(tsne_df, kmeans_tsne.labels_, metric='cosine')
    scores_df.loc['t-SNe', 'Data Size'] = sum(clusters_tsne.memory_usage(deep=True))/1000000

    print('---TSNE---')
    print('KMeans Scaled Silhouette Score: {}'.format(scores_df.loc['t-SNe', 'Silhouette']))
    print('Hopkins Score: ',scores_df.loc['t-SNe', 'Hopkins'])
    print('Data Size: ',scores_df.loc['t-SNe', 'Data Size'])
    acc_score(files_df,count,labels_tsne,'t-SNe')
    return scores_df

def pca_tsne_test(files_df,count,base_df,scores_df):
    # PCA
    pca_tsne = PCA(n_components=20)
    pca_tsne_trans = pca_tsne.fit_transform(base_df)
    pca_tsne_df = pd.DataFrame(pca_tsne_trans)

    # TSNE + PCA
    tsne_pca = TSNE(n_components=3, verbose=1, perplexity=15, n_iter=2000, learning_rate=200,early_exaggeration=4,metric="cosine",init="pca",random_state=42)
    tsne_pca_results = tsne_pca.fit_transform(pca_tsne_df)
    tsne_pca_df = pd.DataFrame(tsne_pca_results, columns=['tsne1', 'tsne2', 'tsne3'])

    
    kmeans_tsne_pca = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(tsne_pca_df)
    labels_tsne_pca = kmeans_tsne_pca.labels_
    clusters_tsne_pca = pd.concat([tsne_pca_df, pd.DataFrame({'tsne_pca_clusters':labels_tsne_pca})], axis=1)

    scores_df.loc['PCA + t-SNe', 'Hopkins'] = hopkins(tsne_pca_df)
    scores_df.loc['PCA + t-SNe', 'Silhouette'] = silhouette_score(tsne_pca_df, kmeans_tsne_pca.labels_, metric='cosine')
    scores_df.loc['PCA + t-SNe', 'Data Size'] = sum(clusters_tsne_pca.memory_usage(deep=True))/1000000
    print('KMeans Scaled Silhouette Score: {}'.format(scores_df.loc['PCA + t-SNe', 'Silhouette']))
    print('Hopkins Score: ',scores_df.loc['PCA + t-SNe', 'Hopkins'])
    print('Data Size: ',scores_df.loc['PCA + t-SNe', 'Data Size'])
    acc_score(files_df,count,labels_tsne_pca,'PCA + t-SNe')

    return scores_df

def acc_score(df,count,preds,model):
    # from https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/

    df['cluster'] = preds
    df = df[['filename','cluster']]
    df = df.merge(count, on="filename")

    cats = df.cluster.max()+1
    df['count_bins'] = pd.qcut(x=df['obj_count'], q=cats,labels=range(0,cats))
    cm = confusion_matrix(df['count_bins'], df['cluster'])

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
      
    scores_df.loc[model, 'Acc'] = np.trace(cm2) / np.sum(cm2)
    scores_df.loc[model, 'Rand'] = rand_score(df['count_bins'], df['cluster'])
    scores_df.loc[model, 'Adj Rand'] = adjusted_rand_score(df['count_bins'], df['cluster'])
    print("Accuracy Score: ",scores_df.loc[model, 'Acc'])
    print("Rand Score: ",scores_df.loc[model, 'Rand'])
    print("Adj Rand Score: ",scores_df.loc[model, 'Adj Rand'])

def main(args):
    """
    Main testing script that will initialize all models and run inference on a raw test set
    """

    # # process arguments
    # args = parse_arguments(args)

    # # check if ZOO is given
    # if not args.ZOO:
    #     raise RuntimeError('Please specify the path to your ZOO directory by using: --ZOO=/path/to/ZOO')

    # # check if bags is given
    # if not args.bags:
    #     raise RuntimeError('Please specify the path to your bag directory by using: --bags=/path/to/bags')

    # # get the data file
    # bag_path = args.bags

    # # get all files from the bag path
    # bags = [bag_path + '/' + f for f in listdir(bag_path) if isfile(join(bag_path, f))]

    # # initialize a detection model
    # model = RetinaNet(args.ZOO + '/RetinaNet/flowers/flower_detector_v1.pth')


    input_path = "data/minneapple/train/images"
    files = os.listdir(input_path)
    files_df = pd.read_csv (r'data/minneapple/vectors/res18_vector_matrix_train_filenames.csv')

    count = pd.read_csv("data/minneapple/count/count.csv",header=None)
    count.columns = ['filename','obj_count']

    vec_mat = np.load('data/minneapple/vectors/res18_vector_matrix_train.npy')
    df = pd.read_csv (r'data/minneapple/vectors/res18_vector_matrix_train_filenames.csv')   
    base_df = pd.DataFrame(data=vec_mat)

    d = {'Model':["Baseline","PCA","t-SNe","PCA + t-SNe"],'Hopkins':[0,0,0,0],'Silhouette':[0,0,0,0],'Acc':[0,0,0,0],'Rand':[0,0,0,0],'Adj Rand':[0,0,0,0],'Data Size':[0,0,0,0]}
    scores_df = pd.DataFrame(data=d)
    scores_df.set_index('Model',inplace=True)

    scores_df = baseline_test(files_df.copy(),count,base_df,scores_df)
    scores_df = pca_test(files_df.copy(),count,base_df,scores_df)
    scores_df = tsne_test(files_df.copy(),count,base_df,scores_df)
    scores_df = pca_tsne_test(files_df.copy(),count,base_df,scores_df)

    print("---SUMMARY---")
    print(scores_df)

if __name__ == "__main__":
    main(sys.argv[1:])