
import argparse
import requests
import json
import tqdm


# Parse arguments
parser = argparse.ArgumentParser(description='Download and convert BIG-bench List Functions data split to jsonl inputs compatible with models.')
parser.add_argument('output_path')
parser.add_argument('--nb_examples', type=int, default=3, help='Number of examples to generate for each task.')
parser.add_argument('--is_algo', action='store_true', help='Whether to generate algorithmic tasks.')
args = parser.parse_args()


# URL helper function
def get_url(i : int):
    str_i = str(i)
    if len(str_i) == 1:
        str_i = "00" + str_i
    elif len(str_i) == 2:
        str_i = "0" + str_i

    url = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/list_functions/c{str_i}/task.json"
    return url


MIN_TASK = 1
MAX_TASK = 250
tasks_ids = list(range(MIN_TASK, MAX_TASK + 1))
nb_examples = args.nb_examples
is_algo = args.is_algo

samples = []
for i in tqdm.tqdm(tasks_ids):
    # Read raw data
    url = get_url(i)
    r = requests.get(url)
    data = json.loads(r.text)

    # Create symbolic data
    sample_input = []
    sample_input.append({
        "role": "system",
        "content": 
            "Your task is to write down the python function responsible for the transformation of the list in the following examples. The format is [input] -> [output]:" 
            if is_algo else
            "Apply a function to the final input list to generate the output list. Use any preceding inputs and outputs as examples to find what is the function used. All example outputs have been generated using the same function."
    })
    for i in range(nb_examples):
        sample_input.append({
            "role": "user",
            "content": f"{str(data['examples'][i]['input'])} -> {str(data['examples'][i]['target'])}"
        })

    sample_input.append({
        "role": "user",
        "content": 
            f"Write the function. Next, write a line to print the output of this function for the input {str(data['examples'][nb_examples]['input'])}:\n```python\n"
            if is_algo else
            f"{str(data['examples'][nb_examples]['input'])} -> "
    })
    samples.append({
        "input": sample_input,
        "ideal": str(data['examples'][nb_examples]['target'])
    })


# Write to output file
with open(args.output_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
print("Saved dataset to: ", args.output_path)


