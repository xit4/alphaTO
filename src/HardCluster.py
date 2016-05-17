from sklearn import cluster
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, help='directory containing csv file')
parser.add_argument('-o', '--outputdir', required=True, help='the output directory the results will be stored in '
                                                             '(/ at the end)')

args = parser.parse_args()

inputfile = args.inputfile
outputdir = args.outputdir

# read the rows from the CSV file. Skip some in skipfooter to reduce computational times/memory requirements
df = pd.read_csv(inputfile, sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
data = df[df.columns[4:]].values


# Spectral Clustering
n_clusters_range = [5, 8, 10, 15]

for n_clusters in n_clusters_range:
    model = cluster.SpectralClustering(n_clusters=n_clusters)
    start = time.time()
    # fit the date and compute compute the clusters
    predicted = model.fit_predict(data)
    end = time.time()
    print('Spectral Clustering execution time ', end-start)
    df.insert(1, 'cluster', predicted)
    df.to_csv(outputdir + '{0}nclusters{1}.csv'.format(outputdir.split('/')[-2], n_clusters), index=False)

    df.drop('cluster', axis=1, inplace=True)

    print(' n_clusters=', n_clusters)

# ----------------------------------
# KMeans
# n_clusters_range = [10, 100, 200, 300, 500, 600, 700]
#
# for n_clusters in n_clusters_range:
#     model = cluster.KMeans(n_clusters=n_clusters)
#     start = time.time()
#     # fit the date and compute compute the clusters
#     predicted = model.fit_predict(data)
#     end = time.time()
#     print('KMeans execution time ', end-start)
#     df.insert(1, 'cluster', predicted)
#     df.to_csv(outputdir + '{0}nclusters{1}.csv'.format(outputdir.split('/')[-2], n_clusters), index=False)
#
#     df.drop('cluster', axis=1, inplace=True)
#
#     print(' n_clusters=', n_clusters)

# ----------------------------------
# DBSCAN
# eps_range = [0.1, 0.5, 1, 2, 3]
# min_sample_range = range(2, 7)
#
# for eps in eps_range:
#     for min_sample in min_sample_range:
#         model = cluster.DBSCAN(eps=eps, min_samples=min_sample)
#         start = time.time()
#         # fit the date and compute compute the clusters
#         predicted = model.fit_predict(data)
#         end = time.time()
#         print('Spectral Clustering execution time ', end-start)
#         df.insert(1, 'cluster', predicted)
#         df.to_csv(outputdir + '{0}eps{1}minsamples{2}.csv'.format(outputdir.split('/')[-1], eps, min_sample),
#                   index=False)
#
#         count = 1
#         for row in df['cluster']:
#             if row == -1:
#                 count += 1
#         df.drop('cluster', axis=1, inplace=True)
#
#         print('Number of outliers ', count, ' eps=', eps, ' min_samples=', min_sample)
#         print('Total number of clusters ', max(predicted))
