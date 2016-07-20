import requests
import argparse
import time
import os

# Script to get upload times from virustotal

url = "https://www.virustotal.com/vtapi/v2/file/report"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-o', '--outputfile', default='./timestamps.csv', help='the output file the times will be stored in')


args = parser.parse_args()

inputdir = args.inputdir
outputfile = args.outputfile

# get the list of files that need to be queried to virustotal
file_list = [item for item in os.listdir(inputdir)
                        if (os.path.isfile(os.path.join(inputdir, item)) and item.endswith('.json'))]

# remove the json extension from each filename
for i, item in enumerate(file_list):
    file_list[i] = item.replace('.json', '')

# start generating the output csv file by inserting the header
with open(outputfile, "w") as text_file:
    print("sha, scan_date", file=text_file)

top_bound = len(file_list)
bound_counter = 0
# cycle through the file list by steps of 4 (virustotal query limit per minute is 4)
for i in range(0, top_bound, 4):
    bound_counter += 4
    if bound_counter > top_bound:
        break

    # setup the API call parameters
    parameters = {"resource": file_list[i] + ', ' + file_list[i+1] + ', ' + file_list[i+2] + ', ' + file_list[i+3],
                  "apikey": "REDACTED"}  # insert API Key instead of REDACTED
    # place the API call
    response = requests.post(url, parameters)
    # parse the response
    json = response.json()
    for response in json:
        with open(outputfile, "a") as text_file:
            print(response['sha256'] + ', ' + response['scan_date'], file=text_file)

    # sleep for a minute to exceed the virustotal cooldown timer
    time.sleep(60)


