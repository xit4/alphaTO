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
from sklearn.neighbors import kneighbors_graph

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
dflbl = pd.read_csv(inputdir + 'parsed{}LabelsClean.csv'.format(tof), sep=',', header=0, engine='python',  skipfooter=0)
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

# scale the data for each feature
# data23_scale = preprocessing.scale(data23)
# data33_scale = preprocessing.scale(data33)
# data40_scale = preprocessing.scale(data40)
# data60_scale = preprocessing.scale(data60)
# data100_scale = preprocessing.scale(data100)
datalbl_scale = preprocessing.scale(datalbl)
data_range = [(datalbl_scale, 'lbl')]#, (data33_scale, '33k'), (data23_scale, '23k'), (data40_scale, '40k'), (data60_scale, '60k'), (data100_scale, '100k')]

for data, label in data_range:
    # KMEANS
    # start = time.time()
    n_clusters_range = range(2, 200)
    max_iter_range = [50, 100] #, 200]#, 300, 400]
    n_init_range = [10, 50] #, 100]

    fig = plt.figure(figsize=(24, 13.5))
    plt.xlim(n_clusters_range[0], n_clusters_range[-1])
    plt.ylim(0.0, 1.0)
    for max_iter in max_iter_range:
        for n_init in n_init_range:
            ARI = []
            Silhouette = []
            for n_clusters in n_clusters_range:
                model = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
                start = time.time()
                # fit the date and compute compute the clusters
                predicted = model.fit_predict(data)
                end = time.time()
                print('KMeans (n_clusters={}) in {}s'.format(n_clusters, end-start))

                ARI.append(metrics.adjusted_rand_score(dflbl['label'].values, predicted))
                print("Adjusted Rand Index: {}".format(ARI[-1]))
                # Silhouette.append(metrics.silhouette_score(data, predicted))
                # print("Silhouette score: {}".format(Silhouette[-1]))

            plt.plot(n_clusters_range, ARI, label="ARI-max{}init{}".format(max_iter, n_init))
            # plt.plot(n_clusters_range, Silhouette, label="Silhouette")

    plt.legend()
    plt.savefig(outputdir + 'KMeansMetrics{}full'.format(label, max_iter, n_init), dpi=80, pad_inches='tight')
    plt.close(fig)

    # end = time.time()
    # print('KMeans Execution time'.format(n_clusters), end-start)