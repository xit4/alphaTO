import pandas as pd
import numpy as np
from sklearn import cluster
import time
import matplotlib
import argparse

# Script plotting how statistical feature change within a specific cluster

# set plotting diplay to Agg when on server
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use an unsupervised model to create clusters
# extract information from the clusters

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, help='CSV file containing the parsed information')
parser.add_argument('-o', '--outputdir', default='./', help='the output directory where the results will be stored in '
                                                            '(/ at the end)')
parser.add_argument('-c', '--cluster-number', type=int, default=200, help='Number of cluster to be fed as parameter to KMeans')

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
cutoff_value = 10 #int(tenth) if tenth > 10 else 10
i = cutoff_value
for label, count in cluster_count.iteritems():
    fig = plt.figure(figsize=(24, 13.5))
    if i == 0:
        break
    i -= 1
    for feature in df.columns.values[5:]:
        plt.title('features over distance in cluster ' + str(label) + ' (' + str(count) + ')')
        plt.xlabel('distance from cluster center')
        plt.ylabel('values for features')
        single_cluster = df.loc[df['cluster'] == label]
        ind = np.argsort(transformed[:, label])[:count]
        plt.text(transformed[ind[0], label], df[feature][ind[0]]+0.1, feature)
        plt.xlim([transformed[ind[0], label], transformed[ind[-1], label]])
        plt.plot(transformed[ind, label], df[feature][ind], marker='o', label=feature)
    plt.legend(loc='upper right' , bbox_to_anchor=(1.1, 1.05))
    plt.savefig(outputdir + 'cluster{0}of{1}'.format(label, n_clusters), dpi=80, pad_inches='tight')
    plt.close(fig)
