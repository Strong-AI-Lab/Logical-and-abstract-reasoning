
import argparse
import json
import os


# Parse arguments
parser = argparse.ArgumentParser(description='Convert ARC data split to (hard, high-dimensional) symbolic jsonl inputs compatible with models. Folders are supported. If folder input, all files in folder are processed.')
parser.add_argument('input_path')
parser.add_argument('output_path')

args = parser.parse_args()
input_file_path = args.input_path
output_file_path = args.output_path

is_folder = os.path.isdir(input_file_path)
if os.path.isdir(output_file_path):
    raise ValueError(f"Output path {output_file_path} must not be a folder.")

print(f"Writing {'all elements of' if is_folder else 'from'} {input_file_path} to {output_file_path} in symbolic format.")

if is_folder:
    input_list = [input_file_path + f for f in os.listdir(input_file_path) if os.path.isfile(input_file_path + f)]
else:
    input_list = [input_file_path]


# Process helper function
def process_file(file_path):
    with open(file_path) as f:
        input_data = json.load(f)

    sample_input = []
    sample_input.append({
        "role": "system",
        "content": "Figure out the pattern in the following examples and apply it to the test case. Your answer must follow the format of the examples. "
    })
    for example in input_data["train"]:
        sample_input.append({
            "role": "system",
            "content": f"{str(example['input'])} -> {str(example['output'])}"
        })

    samples = []
    for test_case in input_data["test"]:
        new_sample = sample_input.copy()
        new_sample.append({
            "role": "system",
            "content": f"{str(test_case['input'])} -> "
        })
        samples.append({
            "input": new_sample,
            "ideal": str(test_case["output"])
        })

    return samples


# Process each file
samples = []
for file_path in input_list:
    print(f"Processing {file_path}...")
    samples += process_file(file_path)


# Write to output file
with open(output_file_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
print("Saved dataset to: ", output_file_path)
