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

# Script to generate plot for various visualization techniques

# set plotting diplay to Agg when on server
matplotlib.use('Agg')

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


# scale the data for each feature
data23_scale = preprocessing.scale(data23)
data33_scale = preprocessing.scale(data33)
data40_scale = preprocessing.scale(data40)
data60_scale = preprocessing.scale(data60)
data100_scale = preprocessing.scale(data100)
data_range = [(data33_scale, '33k'), (data23_scale, '23k'), (data40_scale, '40k'), (data60_scale, '60k'), (data100_scale, '100k')]

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

# Swap in the following lines instead of the previous ones to run different dimensionality reduction models

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