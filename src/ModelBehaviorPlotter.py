import pandas as pd
import matplotlib
import argparse
import numpy as np
from sklearn import cluster
import time
from io import StringIO

# Script used to plot the behaviour of a model when his parameters are changed
# change model code and parameter ranges manually

# set plotting diplay to Agg when on server
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, help='CSV file containing the features')
parser.add_argument('-o', '--outputdir', default='./', help='the output directory where the plot and computation '
                                                            'results will be stored in (/ at the end')


args = parser.parse_args()
inputfile = args.inputfile
outputdir = args.outputdir
output_string = 'eps,minsamples,time,clusters,outliers\n'

df = pd.read_csv(inputfile, sep=',', header=0, engine='python', skipfooter=0)

data = df[df.columns[4:]]

eps_range = np.arange(0.5, 3.25, 0.25)  # [0.1, 0.5, 1, 2 , 3]
min_sample_range = [2, 3, 4, 5, 6]

for eps in eps_range:
    for min_sample in min_sample_range:
        model = cluster.DBSCAN(eps=eps, min_samples=min_sample)
        start = time.time()
        # fit the date and compute compute the clusters
        predicted = model.fit_predict(data)
        end = time.time()

        count = 1
        for row in predicted:
            if row == -1:
                count += 1
        output_string += '{0},{1},{2:.2f},{3},{4}\n'.format(eps, min_sample, end-start, max(predicted), count)

plotting_data = StringIO(output_string)
outdf = pd.read_csv(plotting_data, sep=',')

for feature in ['clusters', 'outliers', 'time']:
    fig = plt.figure()
    for min_sample in min_sample_range:
        plt.xlabel('eps')
        plt.ylabel(feature)
        plt.text(eps_range[0], outdf[feature][min_sample-2], min_sample)
        plt.plot(eps_range, outdf[feature][min_sample-2::5], '-ro')
    plt.savefig(outputdir + feature + 'Behavior', dpi=125, pad_inches='tight')
    plt.close(fig)

