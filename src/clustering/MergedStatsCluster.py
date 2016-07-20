from sklearn import cluster
import time
import pandas as pd
import numpy as np

# MergedStatsCluster.py uses both statistics and permission + domains to computer the clusters
# read the rows from the CSV files. Skip some in skipfooter to reduce computational times/memory requirements
df = pd.read_csv('../CSV/parsedstats.csv', sep=',', header=0, engine='python',  skipfooter=0)
df2 = pd.read_csv('../CSV/parsed.csv', sep=',', header=0, engine='python',  skipfooter=0)

# skip sha, name, certificate and package
temp = df2[df2.columns[4:]].values
data = np.hstack((temp, df2[df2.columns[4:]].values))

eps_range = np.linspace(0.1, 3, 6)
min_sample_range = range(7)

for eps in eps_range:
    for min_sample in min_sample_range:
        model = cluster.DBSCAN(eps=eps, min_samples=min_sample)
        start = time.time()
        # fit the date and compute compute the clusters
        predicted = model.fit_predict(data)
        end = time.time()
        print('DBSCAN execution time ', end-start)
        df2.insert(1, 'cluster', predicted)
        df2.to_csv('../CSV/DBSCANeps{0}minsample{1}.csv'.format(eps, min_sample), index=False)

        count = 1
        for row in df['cluster']:
            if row == -1:
                count += 1
        df.drop('cluster', axis=1, inplace=True)

        print('Number of outliers ', count, ' eps=', eps, ' min_samples=', min_sample)
        print('Total number of clusters ', max(predicted))
