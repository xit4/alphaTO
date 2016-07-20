import pandas as pd
import matplotlib
import argparse

# set plotting diplay to Agg when on server
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Open csv files containing features a clusterss
# build histograms out of those information

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, help='CSV file containing the parsed information')
parser.add_argument('-o', '--outputfile', default='./', help='the output file where the histogram will be stored in ')
parser.add_argument('-c', '--clip', default='', help='integer threshold to disregard bins')


args = parser.parse_args()
inputfile = args.inputfile
outputfile = args.outputfile

# read from a csv containing the clustering results
df = pd.read_csv(inputfile, sep=',', header=0, engine='python',  skipfooter=0)

# skip the columns that do not contain relevant features
X = df[df.columns[5:]]
# select the clusters column to be the label for each bin in the histogram
y = df[df.columns[1]]

plt.hist(y, range(max(y)))
plt.show()


