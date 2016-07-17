import json
import os
import pandas as pd
import numpy as np
import time
import argparse

# get labels from a csv file and generate a unique number for each of them
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True)
parser.add_argument('-o', '--outputfile', default='./parsedlabels.csv', help='the output file the results will be stored in')

inputfile = ''

args = parser.parse_args()

inputfile = args.inputfile
outputfile = args.outputfile


df = pd.read_csv(inputfile, sep=',', header=0, engine='python',  skipfooter=0)
labels = []
labelname = ""
labelvalue = 0
for label in df['labelnames']:
    if label != labelname:
        labelvalue += 1
        labelname = label
    labels.append(labelvalue)

df['label']=labels
# print the results in a CSV file
df.to_csv(outputfile, index=False)