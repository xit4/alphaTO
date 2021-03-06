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
from sklearn import metrics

# Script to perform clustering on multiple datasets by means of multiple models
# the models have to be swapped in manually by uncommenting/commenting the right lines.
# set plotting display to Agg when on server
matplotlib.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True)
parser.add_argument('-o', '--outputdir', required=True, help='the output directory the results will be stored in '
                                                             '(/ at the end)')
parser.add_argument('-t', '--typeoffeature', required=True, default="stats")

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
dflbl = pd.read_csv(inputdir + 'parsed{}Unlabelled.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
end = time.time()
print('Execution time for reading CSVs', end-start)

# skip sha, name, certificate and package (columns not containing relevant features)
data23 = df23[df23.columns[4:]].astype(float).values
data33 = df33[df33.columns[4:]].astype(float).values
data40 = df40[df40.columns[4:]].astype(float).values
data60 = df60[df60.columns[4:]].astype(float).values
data100 = df[df.columns[4:]].astype(float).values
datalbl = dflbl[dflbl.columns[4:]].astype(float).values
data_range = [(datalbl, 'lbl'),(data33, '33k'), (data23, '23k'), (data40, '40k'), (data60, '60k'), (data100, '100k')]

# compute and print the variance for each feature
print("Features: {}".format(dflbl.columns[4:].values))
selector = feature_selection.VarianceThreshold()
for data, label in data_range:
    selector.fit_transform(data)
    print("Variance ({}): {}".format(label, selector.variances_))

# scale the data for each feature
data23_scale = preprocessing.scale(data23)
data33_scale = preprocessing.scale(data33)
data40_scale = preprocessing.scale(data40)
data60_scale = preprocessing.scale(data60)
data100_scale = preprocessing.scale(data100)
datalbl_scale = preprocessing.scale(datalbl)
data_range = [(datalbl_scale, 'lbl'), (data33_scale, '33k'), (data23_scale, '23k'), (data40_scale, '40k'), (data60_scale, '60k'), (data100_scale, '100k')]

for data, label in data_range:
    # KMEANS
    start = time.time()
    ARI = []
    AMI = []
    Homogeneneity = []
    Completeness = []
    v_score = []

    n_clusters_range = range(2, 20)
    for n_clusters in n_clusters_range:
        model = cluster.KMeans(n_clusters==n_clusters)
        # start = time.time()
        # fit the date and compute compute the clusters
        predicted = model.fit_predict(data)
        # end = time.time()
        print('KMeans (n_clusters={})'.format(n_clusters))

        print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
        ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
        print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
        AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
        print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
        Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
        print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
        Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))
        print("V-Score: {}".format(metrics.v_measure_score(dflbl['label'].values, predicted)))
        v_score.append(metrics.v_measure_score(dflbl['label'].values, predicted))
        print("Silhouette score: {}".format(metrics.silhouette_score(data, predicted)))

    end = time.time()
    print('KMeans Execution time'.format(n_clusters), end-start)

    fig = plt.figure(figsize=(24, 13.5))
    plt.xlim(n_clusters_range[0], n_clusters_range[-1])
    plt.ylim(0.0, 1.0)
    plt.plot(n_clusters_range, ARI, label="ARI")
    plt.plot(n_clusters_range, AMI, label="AMI")
    plt.plot(n_clusters_range, Homogeneneity, label="Homogeneity")
    plt.plot(n_clusters_range, Completeness, label="Completeness")
    plt.plot(n_clusters_range, v_score, label="V-score")

    plt.legend()
    plt.savefig(outputdir + 'KMeansMetrics'.format(label, tof), dpi=80, pad_inches='tight')
    plt.close(fig)

# SPECTRAL CLUSTERING
# ARI = []
# AMI = []
# Homogeneneity = []
# Completeness = []
#
# n_clusters_range = range(4, 40)
# for n_clusters in n_clusters_range:
#     model = cluster.SpectralClustering(n_clusters)
#     start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     end = time.time()
#     print('SpectralClustering (n_clusters={}) Execution time '.format(n_clusters), end-start)
#
#     print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
#     ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
#     print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
#     AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
#     print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
#     Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
#     print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
#     Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))
#
# fig = plt.figure(figsize=(24, 13.5))
# plt.xlim(4, 40)
# plt.plot(n_clusters_range, ARI, label="ARI")
# plt.plot(n_clusters_range, AMI, label="AMI")
# plt.plot(n_clusters_range, Homogeneneity, label="Homogeneity")
# plt.plot(n_clusters_range, Completeness, label="Completeness")
# plt.legend()
# plt.savefig(outputdir + 'SpectralClusteringMetrics'.format(label, tof), dpi=80, pad_inches='tight')
# plt.close(fig)

# AGGLOMERATIVE CLUSTERING
# ARI = []
# AMI = []
# Homogeneneity = []
# Completeness = []
# knn_graph = kneighbors_graph(data, 30, include_self=False)
# start = time.time()
# n_clusters_range = range(4, 30)
# for n_clusters in n_clusters_range:
#     model = cluster.AgglomerativeClustering(n_clusters, linkage="ward")
#     # start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     # end = time.time()
#     # print('AgglomerativeClustering (n_clusters={}) Execution time '.format(n_clusters), end-start)
#
#     print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
#     ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
#     # print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
#     # AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
#     # print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
#     # Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
#     # print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
#     # Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))
#
# end = time.time()
# print('AgglomerativeClustering Execution time ', end-start)
# fig = plt.figure(figsize=(24, 13.5))
# plt.xlim(4, 40)
# plt.plot(n_clusters_range, ARI, label="ARI")
# # plt.plot(n_clusters_range, AMI, label="AMI")
# # plt.plot(n_clusters_range, Homogeneneity, label="Homogeneity")
# # plt.plot(n_clusters_range, Completeness, label="Completeness")
# plt.legend()
# plt.savefig(outputdir + 'AgglomerativeClusteringMetrics'.format(label, tof), dpi=80, pad_inches='tight')
# plt.close(fig)

# DBSCAN
# ARI = []
# AMI = []
# Homogeneneity = []
# Completeness = []
# start = time.time()
# eps_range = np.arange(0.1, 3, 0.2)
# for eps in eps_range:
#     model = cluster.DBSCAN(eps=eps)
#     # start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     # end = time.time()
#     # print('DBSCAN (eps={}) Execution time '.format(eps), end-start)
#
#     print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
#     ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
#     print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
#     AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
#     print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
#     Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
#     print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
#     Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))
#
# end = time.time()
# print('DBSCAN Execution time ', end-start)
#
# fig = plt.figure(figsize=(24, 13.5))
# plt.xlim(eps_range[0], eps_range[-1])
# plt.plot(eps_range, ARI, label="ARI")
# plt.plot(eps_range, AMI, label="AMI")
# plt.plot(eps_range, Homogeneneity, label="Homogeneity")
# plt.plot(eps_range, Completeness, label="Completeness")
# plt.legend()
# plt.savefig(outputdir + 'DBSCANMetrics'.format(label, tof), dpi=80, pad_inches='tight')
# plt.close(fig)

# AFFINITY PROPAGATION
# ARI = []
# AMI = []
# Homogeneneity = []
# Completeness = []
#
# damp_range = np.linspace(0.5, 1, num=15)
# for damp in damp_range:
#     model = cluster.DBSCAN(eps=damp)
#     start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     end = time.time()
#     print('Affinity Propagation (eps={}) Execution time '.format(damp), end-start)
#
#     print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
#     ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
#     print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
#     AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
#     print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
#     Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
#     print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
#     Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))
#
# fig = plt.figure(figsize=(24, 13.5))
# plt.xlim(0.5, 1)
# plt.plot(damp_range, ARI, label="ARI")
# plt.plot(damp_range, AMI, label="AMI")
# plt.plot(damp_range, Homogeneneity, label="Homogeneity")
# plt.plot(damp_range, Completeness, label="Completeness")
# plt.legend()
# plt.savefig(outputdir + 'AffinityMetrics'.format(label, tof), dpi=80, pad_inches='tight')
# plt.close(fig)
