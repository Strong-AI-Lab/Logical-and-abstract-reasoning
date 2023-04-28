#!/bin/bash

type=$1
if [ -z "$1" ]; then
  echo "Please specify the type of data to process."
  exit 1
fi

python process_data_raven.py raw/center_single/ $type/center_single.jsonl --type $type
python process_data_raven.py raw/distribute_four/ $type/distribute_four.jsonl --type $type
python process_data_raven.py raw/distribute_nine/ $type/distribute_nine.jsonl --type $type
python process_data_raven.py raw/in_center_single_out_center_single/ $type/in_center_single_out_center_single.jsonl --type $type
python process_data_raven.py raw/in_distribute_four_out_center_single/ $type/in_distribute_four_out_center_single.jsonl --type $type
python process_data_raven.py raw/left_center_single_right_center_single/ $type/left_center_single_right_center_single.jsonl --type $type
python process_data_raven.py raw/up_center_single_down_center_single/ $type/up_center_single_down_center_single.jsonl --type $type