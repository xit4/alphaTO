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
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True)
parser.add_argument('-o', '--outputdir', required=True, help='the output directory the results will be stored in '
                                                             '(/ at the end)')
parser.add_argument('-t', '--typeoffeature', required=True, default="stats")
parser.add_argument('-v', '--verbose', help='set this flag for a verbose script', default=False, action='store_true')
parser.add_argument('-c', '--preclusters', default=12,)

args = parser.parse_args()

inputdir = args.inputdir
outputdir = args.outputdir
tof = args.typeoffeature
verbose = args.verbose
n_clusters = int(args.preclusters)

verboseprint = print if verbose else lambda *a, **k: None

# read the rows from the CSV file. Skip some in skipfooter to reduce computational times/memory requirements
# start = time.time()
dflbl = pd.read_csv(inputdir + 'parsed{}Labels.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
# end = time.time()
# print('Execution time for reading CSVs', end-start)

# skip sha, name, labels and labelnames
datalbl = dflbl[dflbl.columns[4:]].astype(float).values

# scale the data for each feature
datalbl_scaled = preprocessing.scale(datalbl)

# precluster with kmeans to reduce the amount of work
model = cluster.KMeans(n_clusters)
start = time.time()

# fit the data and compute compute the clusters
predicted = model.fit_predict(datalbl_scaled)
end = time.time()
print('Preclustered {} samples with KMeans (n_clusters={}) Execution time {:.2f}s. ARI: {:.2f}'.format(len(datalbl_scaled), n_clusters, end-start, metrics.adjusted_rand_score(dflbl['label'].values, predicted)))

dflbl.insert(1, 'preclusters', predicted)
dflbl.insert(2, 'clusters', 0)
df_range = []
for prediction in range(n_clusters):
    df_range.append(dflbl.loc[dflbl['preclusters'] == prediction])


kmeans = cluster.KMeans()
n_clusters_range = range(2, 20)
dbscan = cluster.DBSCAN()
eps_range = np.arange(0.2, 3.2, 0.2)
affinity = cluster.AffinityPropagation()
damping_range = np.arange(0.5, 1.0, 0.1)
ward = cluster.AgglomerativeClustering(linkage='ward')
complete = cluster.AgglomerativeClustering(linkage='complete')
average = cluster.AgglomerativeClustering(linkage='average')
clustering_names = [
    "KMeans",
    # "DBSCAN",
    # "Affinity",
    # "Ward",
    # "Complete",
    # "Average"
]
clustering_algorithms = [
    kmeans,
    # dbscan,
    # affinity,
    # ward,
    # complete,
    # average
]
clustering_parameters = [
    n_clusters_range,
    # eps_range,
    # damping_range,
    # n_clusters_range,
    # n_clusters_range,
    # n_clusters_range
]

for name, algorithm, parameter_range in zip(clustering_names, clustering_algorithms, clustering_parameters):
    max_predicted = 0
    precluster_counter = 0
    best_counter = 0
    start = time.time()
    for df in df_range:
        data = df[dflbl.columns[6:]].values

        ARI = []
        best_ARI = -1.0
        AMI = []
        Homogeneneity = []
        Completeness = []
        best_predicted = []
        best_parameter = 0
        if len(data) > parameter_range[-1]*2:
            verboseprint("Starting multiple {} instances on precluster #{} with {} sample(s)".format(name, precluster_counter, len(data)))
            for parameter in parameter_range:
                model = algorithm
                if name == "KMeans":
                    algorithm.set_params(n_clusters=parameter, n_init=100)
                elif name == "DBSCAN":
                    algorithm.set_params(eps=parameter)
                elif name == "Affinity":
                    algorithm.set_params(damping=parameter)
                elif name == "Ward":
                    algorithm.set_params(n_clusters=parameter)
                elif name == "Complete":
                    algorithm.set_params(n_clusters=parameter)
                elif name == "Average":
                    algorithm.set_params(n_clusters=parameter)
                # fit the date and compute compute the clusters
                predicted = model.fit_predict(data)


                for x in predicted:
                    x += 1

                current_ARI = metrics.adjusted_rand_score(df['label'].values, predicted)
                if current_ARI > best_ARI:
                    best_ARI = current_ARI
                    best_predicted = predicted
                    best_parameter = parameter

                    best_counter = precluster_counter
                # print('{}({}) Execution time {:.2f}s. ARI: {:.2f}'.format(name, parameter, end-start, current_ARI))

                # print("Adjusted Rand Index: {}".format(metrics.adjusted_rand_score(df['label'].values, predicted)))
                ARI.append(metrics.adjusted_rand_score(df['label'].values, predicted))
                # print("Adjusted Mutual Information: {}".format(metrics.adjusted_mutual_info_score(df['label'].values, predicted)))
                AMI.append(metrics.adjusted_mutual_info_score(df['label'].values, predicted))
                # print("Homogeneity: {}".format(metrics.homogeneity_score(df['label'].values, predicted)))
                Homogeneneity.append(metrics.homogeneity_score(df['label'].values, predicted))
                # print("Completeness: {}".format(metrics.completeness_score(df['label'].values, predicted)))
                Completeness.append(metrics.completeness_score(df['label'].values, predicted))
                verboseprint('\r' + '{0:.2f}% completed '.format((parameter-parameter_range[0])/(parameter_range[-1]-parameter_range[0]+1)*100), end='', flush=True)
            verboseprint('\rBest run: {}({})  ARI={:.2f}'.format(name, best_parameter, best_ARI))
        else:
            verboseprint("Could not start {} on precluster #{} with {} sample(s) (not enough samples)".format(name, precluster_counter, len(data)))
            best_predicted = df['preclusters']

        best_predicted += max_predicted - np.min(best_predicted)
        max_predicted = np.max(best_predicted) + 1
        dflbl.loc[df.index, 'clusters'] = best_predicted
        precluster_counter += 1
        # df.loc[df.index, 'clusters'] = best_predicted
        # df.to_csv(outputdir + 'PRECLUSTERING.csv', index=False)

        # fig = plt.figure(figsize=(24, 13.5))
        # plt.xlim(n_clusters_range[0], n_clusters_range[-1])
        # plt.ylim(0, 1)
        # plt.plot(n_clusters_range, ARI, label="ARI")
        # plt.plot(n_clusters_range, AMI, label="AMI")
        # plt.plot(n_clusters_range, Homogeneneity, label="Homogeneity")
        # plt.plot(n_clusters_range, Completeness, label="Completeness")
        # plt.legend()
        # plt.show()
        # plt.savefig(outputdir + 'KMeansMetrics'.format(label, tof), dpi=80, pad_inches='tight')
        # plt.close(fig)
    end = time.time()
    print('{} Preclustering ARI={:.2f}  Final ARI={:.2f} time={:.2f}s'.format(name,
                                                                             metrics.adjusted_rand_score(dflbl['label'].values, dflbl['preclusters'].values),
                                                                             metrics.adjusted_rand_score(dflbl['label'].values, dflbl['clusters'].values),
                                                                             end-start
                                                                             ))
    dflbl.to_csv(outputdir + 'PRECLUSTERING{}.csv'.format(name), index=False)

    # pca = decomposition.PCA(n_components=3)
    # pca_reduced = pca.fit_transform(datalbl_scaled)
    #
    # fig = plt.figure(figsize=(24, 13.5))
    # ax = fig.add_subplot(221, projection='3d')
    # ax.scatter(xs=pca_reduced[:, 0], ys=pca_reduced[:, 1], zs=pca_reduced[:, 2], zdir='z', s=10, c=dflbl['label'].values)
    # ax = fig.add_subplot(222, projection='3d')
    # ax.scatter(xs=pca_reduced[:, 0], ys=pca_reduced[:, 1], zs=pca_reduced[:, 2], zdir='z', s=10, c=dflbl['clusters'].values)
    # ax = fig.add_subplot(223, projection='3d')
    # ax.scatter(xs=pca_reduced[:, 0], ys=pca_reduced[:, 1], zs=pca_reduced[:, 2], zdir='z', s=10, c=dflbl['preclusters'].values)
    # # plt.show()
    # plt.savefig(outputdir + '{}{}{}'.format(name, tof), dpi=80, pad_inches='tight')
    # plt.close(fig)