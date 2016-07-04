from sklearn import cluster
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import manifold
import numpy as np
import matplotlib

# set plotting diplay to Agg when on server
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True)
parser.add_argument('-o', '--outputdir', required=True, help='the output directory the results will be stored in '
                                                             '(/ at the end)')
parser.add_argument('-t', '--typeoffeature', default="stats")

args = parser.parse_args()

inputdir = args.inputdir
outputdir = args.outputdir
tof = args.typeoffeature

# read the rows from the CSV file. Skip some in skipfooter to reduce computational times/memory requirements
start = time.time()
df = pd.read_csv(inputdir + 'parsed{}.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
df33 = pd.read_csv(inputdir + 'parsed{}33.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
df23 = pd.read_csv(inputdir + 'parsed{}23.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
df40 = pd.read_csv(inputdir + 'parsed{}40.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
df60 = pd.read_csv(inputdir + 'parsed{}60.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
end = time.time()
print('Execution time for reading CSVs', end-start)

# skip sha, name, certificate and package
data23 = df23[df23.columns[4:]].astype(float).values
data33 = df33[df33.columns[4:]].astype(float).values
data40 = df40[df40.columns[4:]].astype(float).values
data60 = df60[df60.columns[4:]].astype(float).values
data100 = df[df.columns[4:]].astype(float).values
data_range = [(data33, '33k'), (data23, '23k'), (data40, '40k'), (data60, '60k'), (data100, '100k')]

# compute and print the variance for each feature
print("Features: {}".format(df23.columns[4:].values))
selector = feature_selection.VarianceThreshold()
for data, label in data_range:
    selector.fit_transform(data)
    print("Variance ({}): {}".format(label, selector.variances_))

# normalize the data for each feature
# data23_norm = preprocessing.normalize(data23, axis=0)
# data33_norm = preprocessing.normalize(data33, axis=0)
# data40_norm = preprocessing.normalize(data40, axis=0)
# data60_norm = preprocessing.normalize(data60, axis=0)
# data100_norm = preprocessing.normalize(data100, axis=0)
# data_range = [(data33_norm, '33k'), (data23_norm, '23k'), (data40_norm, '40k'), (data60_norm, '60k'), (data100_norm, '100k')]

# scale the data for each feature
data23_norm = preprocessing.scale(data23)
data33_norm = preprocessing.scale(data33)
data40_norm = preprocessing.scale(data40)
data60_norm = preprocessing.scale(data60)
data100_norm = preprocessing.scale(data100)
data_range = [(data33_norm, '33k'), (data23_norm, '23k'), (data40_norm, '40k'), (data60_norm, '60k'), (data100_norm, '100k')]

model = decomposition.PCA(n_components=3)
modellabel = "pca"
for data, label in data_range:
    # # fit the model to the data
    # # compute the actual reduction. Save elapsed times to compare solutions
    start = time.time()
    model_reduced = model.fit_transform(data)
    end = time.time()
    print('Execution time for {} reduction'.format(modellabel), end-start)

    fig = plt.figure(figsize=(24, 13.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=model_reduced[:, 0], ys=model_reduced[:, 1], zs=model_reduced[:, 2], zdir='z', s=10)
    plt.title('{} on {} samples ({} features)'.format(modellabel, label, tof))
    plt.show()
    # plt.savefig(outputdir + '{}{}{}norm'.format(modellabel, label, tof), dpi=80, pad_inches='tight')
    plt.close(fig)

# model = manifold.MDS(n_components=3)
# modellabel = "mds"
# for data, label in data_range:
#     # # fit the model to the data
#     # # compute the actual reduction. Save elapsed times to compare solutions
#     start = time.time()
#     model_reduced = model.fit_transform(data[:10000])
#     end = time.time()
#     print('Execution time for {} reduction'.format(modellabel), end-start)
#
#     fig = plt.figure(figsize=(24, 13.5))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xs=model_reduced[:, 0], ys=model_reduced[:, 1], zs=model_reduced[:, 2], zdir='z', s=10)
#     plt.title('{} on {} samples ({} features)'.format(modellabel, label, tof))
#     # plt.show()
#     plt.savefig(outputdir + '{}{}{}norm'.format(modellabel, label, tof), dpi=80, pad_inches='tight')
#     plt.close(fig)
#
# model = manifold.Isomap(n_components=3)
# modellabel = "isomap"
# for data, label in data_range:
#     # # fit the model to the data
#     # # compute the actual reduction. Save elapsed times to compare solutions
#     start = time.time()
#     model_reduced = model.fit_transform(data[:10000])
#     end = time.time()
#     print('Execution time for {} reduction'.format(modellabel), end-start)
#
#     fig = plt.figure(figsize=(24, 13.5))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xs=model_reduced[:, 0], ys=model_reduced[:, 1], zs=model_reduced[:, 2], zdir='z', s=10)
#     plt.title('{} on {} samples ({} features)'.format(modellabel, label, tof))
#     # plt.show()
#     plt.savefig(outputdir + '{}{}{}norm'.format(modellabel, label, tof), dpi=80, pad_inches='tight')
#     plt.close(fig)

# mds = manifold.MDS(n_components=3)
# # # fit the model to the data
# # compute the actual reduction. Save elapsed times to compare solutions
# start = time.time()
# tsne_reduced = mds.fit_transform(data)
# end = time.time()
# print('Execution time for Isomap reduction', end-start)

# iso = manifold.Isomap(n_neighbors=50, n_components=3)
# # # fit the model to the data
# # compute the actual reduction. Save elapsed times to compare solutions
# start = time.time()
# tsne_reduced = iso.fit_transform(data)
# end = time.time()
# print('Execution time for Isomap reduction', end-start)

# Spectral Clustering
# n_clusters_range = [5, 8, 10, 15]
#
# for n_clusters in n_clusters_range:
#     model = cluster.SpectralClustering(n_clusters=n_clusters)
#     start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     end = time.time()
#     print('Spectral Clustering Execution time ', end-start)
#     df.insert(1, 'cluster', predicted)
#     df.to_csv(outputdir + '{0}nclusters{1}.csv'.format(outputdir.split('/')[-2], n_clusters), index=False)
#
#     df.drop('cluster', axis=1, inplace=True)
#
#     print(' n_clusters=', n_clusters)

# ----------------------------------
# KMeans

# ----------------------------------
# DBSCAN
#
# model = cluster.DBSCAN(eps=3, min_samples=6)
# start = time.time()
# # fit the date and compute compute the clusters
# predicted = model.fit_predict(data)
# end = time.time()
# print('DBSCAN Execution time ', end-start)
#
# count = 1
# for row in predicted:
#     if row == -1:
#         count += 1
#
# print('Number of outliers ', count)
# print('Total number of clusters ', max(predicted))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = (np.linspace(0, 1.0, max(predicted)+1))
#
# #for i, point in enumerate(predicted):
# ax.scatter(xs=pca_reduced[:, 0], ys=tsne_reduced[:, 1], zs=tsne_reduced[:, 2], zdir='z', s=15, c=plt.cm.RdYlBu(colors[predicted]))
# plt.show()
# print()