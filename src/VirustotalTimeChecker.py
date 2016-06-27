import requests
import argparse
import time
import os

url = "https://www.virustotal.com/vtapi/v2/file/report"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', required=True, help='directory containing json files (/ at the end)')
parser.add_argument('-o', '--outputfile', default='./timestamps.csv', help='the output file the times will be stored in')


args = parser.parse_args()

inputdir = args.inputdir
outputfile = args.outputfile


file_list = [item for item in os.listdir(inputdir)
                        if (os.path.isfile(os.path.join(inputdir, item)) and item.endswith('.json'))]

for i, item in enumerate(file_list):
    file_list[i] = item.replace('.json', '')

with open(outputfile, "w") as text_file:
    print("sha, scan_date", file=text_file)

top_bound = len(file_list)
bound_counter = 0
for i in range(0, top_bound, 4):
    bound_counter += 4
    if bound_counter > top_bound:
        break

    parameters = {"resource": file_list[i] + ', ' + file_list[i+1] + ', ' + file_list[i+2] + ', ' + file_list[i+3],
                  "apikey": "bf91c26b3e19b30b62c0116047f8d4ab8bce355b245e0085ef3b4f115691e7cc"}
    response = requests.post(url, parameters)
    json = response.json()
    for response in json:
        with open(outputfile, "a") as text_file:
            print(response['sha256'] + ', ' + response['scan_date'], file=text_file)

    time.sleep(60)


