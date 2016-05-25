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

df = pd.read_csv(inputfile, sep=',', header=0, engine='python',  skipfooter=0)

X = df[df.columns[5:]]
y = df[df.columns[1]]

plt.hist(y, range(max(y)))
plt.show()


