import pandas as pd
import numpy as np
from sklearn import cluster
import time
import matplotlib
import argparse

# set plotting diplay to Agg when on server
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use an unsupervised model to create clusters
# extract information from the clusters

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, help='CSV file containing the parsed information')
parser.add_argument('-o', '--outputdir', default='./', help='the output directory where the results will be stored in '
                                                            '(/ at the end)')
parser.add_argument('-c', '--cluster-number', default=200, help='Number of cluster to be fed as parameter to KMeans')

args = parser.parse_args()
inputfile = args.inputfile
outputdir = args.outputdir
n_clusters = args.cluster_number

df = pd.read_csv(inputfile, sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
data = df[df.columns[4:]].values

model = cluster.KMeans(n_clusters=n_clusters, random_state=1)
start = time.time()
# fit the date and compute compute the clusters
predicted = model.fit_predict(data)
# transform the model to a distance based model
transformed = model.transform(data)
end = time.time()
print('Model execution time ', end-start)

# insert the cluster column in the dataframe in order to ease future operations
df.insert(4, 'cluster', predicted)

cluster_count = df['cluster'].value_counts()

tenth = n_clusters/10
cutoff_value = int(tenth) if tenth > 10 else 10
# plot the biggest clusters for each feature behavior over distance
for feature in df.columns.values[5:]:
    fig = plt.figure()
    plt.title(feature + ' over distance')
    plt.xlabel('distance from respective center')
    plt.ylabel('number of ' + feature)

    i = cutoff_value
    for label, count in cluster_count.iteritems():
        if i == 0:
            break
        i -= 1
        ind = np.argsort(transformed[:, label])[:count]
        plt.plot(transformed[ind, label], df[feature][ind], marker='o')
    plt.savefig(outputdir + feature + '{0}'.format(n_clusters), dpi=125, pad_inches='tight')
    plt.close(fig)
