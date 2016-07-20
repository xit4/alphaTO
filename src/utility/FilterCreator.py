import json
import os
import pandas as pd
import argparse

# explore jsons to generate two filters splitting the samples in two based on information contained in them

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-o', '--outputdir', help='the output directory the results will be stored in (/ at the end)')

inputdir = ''
outputdir = './'

args = parser.parse_args()

inputdir = args.inputdir
if args.outputdir:
    outputdir = args.outputdir

# find out how many samples we have
numberOfRows = 0
numberOfRows = len([item for item in os.listdir(inputdir)
                    if (os.path.isfile(os.path.join(inputdir, item)) and item.endswith('.json'))])

# columns we are interested in
columnNames = ('sha', 'name', 'package', 'certificate')
# create DataFrame
df = pd.DataFrame(index=range(0, numberOfRows), columns=columnNames)

# output strings
file1 = ''
file2 = ''

# initialize index counter
i = 0
# for each (report.json) file in the directory
for filename in (os.listdir(inputdir)):
    pathname = os.path.join(inputdir, filename)
    if not os.path.isfile(pathname) or not filename.endswith('.json'):
        continue
    # open it
    with open(pathname) as data_file:
        # load the json data inside of it
        data = json.load(data_file)
        # find the domains used by the app

        if data['androguard']['app_name'] == '成人快播':
            file1 += filename + '\n'
        else:
            file2 += filename + '\n'

    i += 1
    # print a status message (mostly to keep ssh connection on... hopefully)
    print('\r' + '{0:.2f}% reports parsed '.format(i/numberOfRows*100), end='', flush=True)

print('')

with open(outputdir+"filterSixty.txt", "w") as text_file:
    text_file.write("%s" % file1)
with open(outputdir+"filterForty.txt", "w") as text_file:
    text_file.write("%s" % file2)