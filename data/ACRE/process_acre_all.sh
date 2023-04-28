#!/bin/bash

type=$1
if [ -z "$1" ]; then
  echo "Please specify the type of data to process."
  exit 1
fi

for folder in IID Comp Sys
do
    for file in train val test
    do
        python process_data_acre.py raw/$folder/$file.json $type/$folder/$file.jsonl --type $type
    done 
done

