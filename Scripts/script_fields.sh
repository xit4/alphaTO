#! /bin/bash

## Author: Andrea Marcelli
## PhD Student
## Email: andrea.marcelli@polito.it

## Script to parse filters/receiver/ in json file
## and to create a CSV file with One-Max Encoding


# Check argument number
if [ "$#" -ne 4 ]
then
    echo "Illegal number of parameters"
    echo "Usage: ./script_fields.sh input_file.txt jq_expression output_file.csv #_of_samples"
    echo "Example Usage: ./script_fields.sh receivers_filtered_300.txt .androguard.receivers[] output_file.csv 100000"
    echo "Example Usage: ./script_fields.sh filters_filtered_300.txt .androguard.filters[] filters_output.csv 100000"
    echo "Example Usage: ./script_fields.sh permissions_filtered_200.txt .androguard.permissions[] permissions_output.csv 100000"
    echo "Note:  It takes the 2nd column from the input file"
    exit
fi



# Build progressbar strings and print the ProgressBar line
# Progress : [########################################] 100%
function ProgressBar {
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")
printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"
}
# Variables
_start=1
_end=$4



### Create a First Line with the name of all the columns
printf "Application Name," >> $3
while read a field
do
    printf %s, $field >> $3
done < $1
printf "\n" >> $3


### Create OneMax Encoding
counter=0
# For each report
for i in *.json
do
  # Print the sha of the File
  name=`echo $i | sed 's/.json//g'`
  printf %s, $name >> $3

  # Update Progress Bar
  ProgressBar ${counter} ${_end}
  counter=$(($counter+1))

  # Save in a temporary file the selected part of the Json file
  cat $i | jq -r $2 > temp_file.txt

  # For each field in the 2nd column of the file
  while read a field
  do
    # Check if it is present in your app
    this_field=`cat temp_file.txt| grep $field`
    if [ -z "$this_field" ]
    then
      this_field=0
    else
      this_field=1
    fi

    printf %s, $this_field >> $3

    #statements
  done < $1

  # Print new line in the output file.
  printf "\n" >> $3

done
# Remove the temporary file
rm temp_file.txt
printf '\nFinished!\n'
