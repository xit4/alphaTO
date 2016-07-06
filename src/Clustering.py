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

# set plotting diplay to Agg when on server
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
# df = pd.read_csv(inputdir + 'parsed{}.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
# df33 = pd.read_csv(inputdir + 'parsed{}33.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
# df23 = pd.read_csv(inputdir + 'parsed{}23.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
# df40 = pd.read_csv(inputdir + 'parsed{}40.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
# df60 = pd.read_csv(inputdir + 'parsed{}60.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
dflbl = pd.read_csv(inputdir + 'parsed{}Labels.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
end = time.time()
print('Execution time for reading CSVs', end-start)

# skip sha, name, certificate and package
# data23 = df23[df23.columns[4:]].astype(float).values
# data33 = df33[df33.columns[4:]].astype(float).values
# data40 = df40[df40.columns[4:]].astype(float).values
# data60 = df60[df60.columns[4:]].astype(float).values
# data100 = df[df.columns[4:]].astype(float).values
datalbl = dflbl[dflbl.columns[4:]].astype(float).values
data_range = [(datalbl, 'lbl')]#,(data33, '33k'), (data23, '23k'), (data40, '40k'), (data60, '60k'), (data100, '100k')]

# compute and print the variance for each feature
print("Features: {}".format(dflbl.columns[4:].values))
selector = feature_selection.VarianceThreshold()
for data, label in data_range:
    selector.fit_transform(data)
    print("Variance ({}): {}".format(label, selector.variances_))

# # normalize the data for each feature
# data23_norm = preprocessing.normalize(data23, axis=0)
# data33_norm = preprocessing.normalize(data33, axis=0)
# data40_norm = preprocessing.normalize(data40, axis=0)
# data60_norm = preprocessing.normalize(data60, axis=0)
# data100_norm = preprocessing.normalize(data100, axis=0)
# data_range = [(data23_norm, '23k'), (data33_norm, '33k'), (data40_norm, '40k'), (data60_norm, '60k'), (data100_norm, '100k')]

# scale the data for each feature
# data23_scale = preprocessing.scale(data23)
# data33_scale = preprocessing.scale(data33)
# data40_scale = preprocessing.scale(data40)
# data60_scale = preprocessing.scale(data60)
# data100_scale = preprocessing.scale(data100)
datalbl_scale = preprocessing.scale(datalbl)
data_range = [(datalbl_scale, 'lbl')]#, (data33_scale, '33k'), (data23_scale, '23k'), (data40_scale, '40k'), (data60_scale, '60k'), (data100_scale, '100k')]

for data, label in data_range:

    # pca = decomposition.PCA(n_components=3)
    # # # fit the model to the data
    # # # compute the actual reduction. Save elapsed times to compare solutions
    # start = time.time()
    # pca_reduced = pca.fit_transform(data)
    # end = time.time()
    # print('Execution time for PCA ({}) reduction'.format(label), end-start)
    #
    # tsne = manifold.TSNE(n_components=3)
    # # fit the model to the data
    # # compute the actual reduction. Save elapsed times to compare solutions
    # start = time.time()
    # tsne_reduced = tsne.fit_transform(data[:10000, :])
    # end = time.time()
    # print('Execution time for TSNE ({}) reduction'.format(label), end-start)

    # KMEANS
    ARI = []
    AMI = []
    Homogeneneity = []
    Completeness = []

    n_clusters_range = range(4, 40)
    for n_clusters in n_clusters_range:
        model = cluster.KMeans(n_clusters, n_init=50)
        start = time.time()
        # fit the date and compute compute the clusters
        predicted = model.fit_predict(data)
        end = time.time()
        print('KMeans (n_clusters={}) Execution time '.format(n_clusters), end-start)

        print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(dflbl['label'].values, predicted)))
        ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
        print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted)))
        AMI.append(metrics.adjusted_mutual_info_score(dflbl['label'].values, predicted))
        print("Homogeneity: {}".format(metrics.homogeneity_score(dflbl['label'].values, predicted)))
        Homogeneneity.append(metrics.homogeneity_score(dflbl['label'].values, predicted))
        print("Completeness: {}".format(metrics.completeness_score(dflbl['label'].values, predicted)))
        Completeness.append(metrics.completeness_score(dflbl['label'].values, predicted))

        fig = plt.figure(figsize=(24, 13.5))
        plt.xlim(n_clusters_range[0], n_clusters_range[-1])
        plt.plot(n_clusters_range, ARI, label="ARI")
        plt.plot(n_clusters_range, AMI, label="AMI")
        plt.plot(n_clusters_range, Homogeneneity, label="Homogeneity")
        plt.plot(n_clusters_range, Completeness, label="Completeness")
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
    #
    # n_clusters_range = range(4, 40)
    # for n_clusters in n_clusters_range:
    #     model = cluster.AgglomerativeClustering(n_clusters)
    #     start = time.time()
    #     # fit the date and compute compute the clusters
    #     predicted = model.fit_predict(data)
    #     end = time.time()
    #     print('AgglomerativeClustering (n_clusters={}) Execution time '.format(n_clusters), end-start)
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
    # plt.savefig(outputdir + 'AgglomerativeClusteringMetrics'.format(label, tof), dpi=80, pad_inches='tight')
    # plt.close(fig)

    # DBSCAN
    # ARI = []
    # AMI = []
    # Homogeneneity = []
    # Completeness = []
    #
    # eps_range = np.linspace(0.5, 2, num=15)
    # for eps in eps_range:
    #     model = cluster.DBSCAN(eps=eps)
    #     start = time.time()
    #     # fit the date and compute compute the clusters
    #     predicted = model.fit_predict(data)
    #     end = time.time()
    #     print('DBSCAN (eps={}) Execution time '.format(eps), end-start)
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
    # plt.xlim(eps_range[0], eps_range[-1])
    # plt.plot(eps_range, ARI, label="ARI")
    # plt.plot(eps_range, AMI, label="AMI")
    # plt.plot(eps_range, Homogeneneity, label="Homogeneity")
    # plt.plot(eps_range, Completeness, label="Completeness")
    # plt.legend()
    # plt.savefig(outputdir + 'DBSCANMetrics'.format(label, tof), dpi=80, pad_inches='tight')
    # plt.close(fig)

    # fig = plt.figure(figsize=(24, 13.5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs=tsne_reduced[:, 0], ys=tsne_reduced[:, 1], zs=tsne_reduced[:, 2], zdir='z', s=10, c=predicted[:10000])
    # plt.title('DBSCAN ({2}) and TSNE on {0} samples ({1} features)'.format(label, tof, 2))
    # # plt.show()
    # plt.savefig(outputdir + 'tsne{}{}normDBSCAN'.format(label, tof), dpi=80, pad_inches='tight')
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(24, 13.5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xs=pca_reduced[:, 0], ys=pca_reduced[:, 1], zs=pca_reduced[:, 2], zdir='z', s=10, c=predicted)
    # plt.title('DBSCAN ({2}) and PCA on {0} samples ({1} features)'.format(label, tof, 2))
    # # plt.show()
    # plt.savefig(outputdir + 'pca{}{}normDBSCAN'.format(label, tof), dpi=80, pad_inches='tight')
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
#     n_clusters_range = [5, 8, 10, 15, 100, 200]
#     for n_clusters in n_clusters_range:
#         model = cluster.KMeans(n_clusters)
#         start = time.time()
#         # fit the date and compute compute the clusters
#         predicted = model.fit_predict(data)
#         end = time.time()
#         print('KMeans Execution time ', end-start)

# ----------------------------------
# DBSCAN
#     eps_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
#     for eps in eps_range:
#         model = cluster.DBSCAN(eps=eps)
#         start = time.time()
#         # fit the date and compute compute the clusters
#         predicted = model.fit_predict(data)
#         end = time.time()
#         print('DBSCAN (eps={}) Execution time '.format(eps), end-start)
