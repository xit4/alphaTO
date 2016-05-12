import numpy as np
from sklearn import manifold, decomposition
from sklearn import cluster
import pylab as pl
import time
import pandas as pd


# extract data from the CSV, skip the first row cause it does not contain actual data and skip the last 50000 lines to
# reduce the number of rows the computation has to process
df = pd.read_csv('../CSV/parsedstats.csv', sep=',', header=0, engine='python',  skipfooter=0)

# for index, row in df.iterrows():
#     if row['name'] == '成人快播':
#         df.drop(index, inplace=True)

data = df[df.columns[4:]].values

# initialize the reduction models with n_components being the number of dimensions we want to reduce the data to
pca = decomposition.PCA(n_components=2)
tsne = manifold.TSNE(n_components=2)

# fit the model to the data
# compute the actual reduction. Save elapsed times to compare solutions
start = time.time()
pca_reduced = pca.fit_transform(data)
end = time.time()
print('execution time for PCA reduction', end-start)
# start = time.time()
# tsne_reduced = tsne.fit_transform(data)
# end = time.time()
# print('execution time for TSNE reduction', end-start)

# ----------------------------
# DBSCAN (uncomment as needed)
# dbscan = cluster.DBSCAN()
# start = time.time()
# # fit the date and compute compute the clusters
# predicted = dbscan.fit_predict(data)
# end = time.time()
# print('DBSCAN execution time without reduction', end-start)
#
# dbscan = cluster.DBSCAN()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedPCA = dbscan.fit_predict(pca_reduced)
# end = time.time()
# print('DBSCAN execution time with PCA reduction', end-start)
#
# dbscan = cluster.DBSCAN()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedTSNE = dbscan.fit_predict(tsne_reduced)
# end = time.time()
# print('DBSCAN execution time with TSNE reduction', end-start)

# ----------------------------
# KMeans (uncomment as needed)
kmeans = cluster.KMeans()
start = time.time()
# fit the date and compute compute the clusters
predicted = kmeans.fit_predict(data)
end = time.time()
print('KMeans execution time without reduction', end-start)
#
# kmeans = cluster.KMeans()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedPCA = kmeans.fit_predict(pca_reduced)
# end = time.time()
# print('KMeans execution time with PCA reduction', end-start)
#
# kmeans = cluster.KMeans()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedTSNE = kmeans.fit_predict(tsne_reduced)
# end = time.time()
# print('KMeans execution time with TSNE reduction', end-start)

# --------------------------------------------
# AgglomerativeClustering (uncomment as needed)
# initialize the model asking for n_clusters
# agglomerativeclustering = cluster.AgglomerativeClustering(n_clusters=30)
# start = time.time()
# # fit the date and compute compute the clusters
# predicted = agglomerativeclustering.fit_predict(data)
# end = time.time()
# print('AgglomerativeClustering execution time without reduction', end-start)
#
# agglomerativeclustering = cluster.AgglomerativeClustering(n_clusters=5)
# start = time.time()
# # fit the date and compute compute the clusters
# predictedPCA = agglomerativeclustering.fit_predict(pca_reduced)
# end = time.time()
# print('AgglomerativeClustering execution time with PCA reduction', end-start)
#
# agglomerativeclustering = cluster.AgglomerativeClustering(n_clusters=5)
# start = time.time()
# # fit the date and compute compute the clusters
# predictedTSNE = agglomerativeclustering.fit_predict(tsne_reduced)
# end = time.time()
# print('AgglomerativeClustering execution time with TSNE reduction', end-start)

# -----------------------------------------
# AffinityPropagation (uncomment as needed)
# affinitypropagation = cluster.AffinityPropagation()
# start = time.time()
# # fit the date and compute compute the clusters
# predicted = affinitypropagation.fit_predict(data)
# end = time.time()
# print('AffinityPropagation execution time without reduction', end-start)

# affinitypropagation = cluster.AffinityPropagation()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedPCA = affinitypropagation.fit_predict(pca_reduced)
# end = time.time()
# print('AffinityPropagation execution time with PCA reduction', end-start)
#
# affinitypropagation = cluster.AffinityPropagation()
# start = time.time()
# # fit the date and compute compute the clusters
# predictedTSNE = affinitypropagation.fit_predict(tsne_reduced)
# end = time.time()
# print('AffinityPropagation execution time with TSNE reduction', end-start)


# df.insert(1, 'cluster', predicted)
#
# df.to_csv('../CSV/clusterized.csv', index=False)

# plot the results
pl.scatter(pca_reduced[:, 0], pca_reduced[:, 1], c=predicted,
           s=75,
           marker='.')
pl.show()
# pl.scatter(pca_reduced[:, 0], pca_reduced[:, 1], c=predictedPCA,
#            s=75,
#            marker='s')
# pl.show()
# pl.scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], c=predicted,
#            s=75,
#            marker='s')
# pl.show()
# pl.scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], c=predicted,
#            s=75,
#            marker='s')
# for i in range(0, len(pca_reduced)):
#     pl.text(tsne_reduced[i, 0], tsne_reduced[i, 1], df[df.columns[2]].values[i])
# pl.show()

