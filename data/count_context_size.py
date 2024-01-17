
import json
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('data_file', type=str)

args = parser.parse_args()
data_file = args.data_file

counts = []
with open(data_file, 'r') as f:
    data = f.readlines()
    for line in data:
        sample = json.loads(line)
        sample = "\n".join([inp['content'] for inp in sample['input']])
        words = re.findall(r'\b\w+\b', sample)
        counts.append(len(words))
   
print(f"Average context size: {sum(counts)/len(counts)}")
        