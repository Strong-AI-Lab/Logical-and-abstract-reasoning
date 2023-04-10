
import argparse
import json


# Parse arguments
parser = argparse.ArgumentParser(description='Convert ACRE data split to symbolic or text jsonl inputs compatible with models.')
parser.add_argument('input_path')
parser.add_argument('output_path')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--symbolic', action='store_true')
group.add_argument('--text', action='store_true')

args = parser.parse_args()
input_file_path = args.input_path
output_file_path = args.output_path
symbolic = args.symbolic

print(f"Writing to {output_file_path} from {input_file_path} in {'symbolic' if symbolic else 'text'} format.")


# Read raw data
with open(input_file_path) as f:
    input_data = json.load(f)

print(f"Read {len(input_data)} samples.")
            


# Symbolic translation functions
def get_instructions_symbolic():
    return {
                "role": "system",
                "content": "Figure out the pattern in the following examples and apply it to the test case. Your answer must follow the format of the examples. You can answer 2 if the solution cannot be determined."
    }

LIGHT_DICT_SYMB = {"on": 2, "off": 0}
def get_example_symbolic(objects, light_state):
    return [
                {
                     "role": "system",
                     "content": f"{str(objects)} -> {LIGHT_DICT_SYMB[light_state]}"
                }
    ]

def get_trial_symbolic(objects, label):
    return [
                {
                    "role": "system",
                    "content": f"{str(objects)} -> "
                }
            ], label


# Text translation functions
LIGHT_DICT_TEXT = ["off", "undetermined", "on"]
with open("object_dict.json", "r") as object_dict_file:
    OBJECT_DICT_TEXT = json.load(object_dict_file)
    
def get_instructions_text():
    return {
                "role": "system",
                "content": "Objects of various color, shape, and texture are displayed. Some objects may contain a device to turn a light on if displayed. From the observations, deduce if the light is on, off, or if the state cannot be determined. Your answer must contain a single word: on, off, or undetermined."
    }

def get_example_text(objects, light_state):
    object_text = ""
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        c = obj_desc["color"]
        s = obj_desc["shape"]
        t = obj_desc["material"]
        object_text += f"A {c} {s} in {t} is visible. "
    return [
                {
                     "role": "system",
                     "content": object_text + f"The light is {light_state}."
                }
    ]
def get_trial_text(objects, label):
    object_text = ""
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        c = obj_desc["color"]
        s = obj_desc["shape"]
        t = obj_desc["material"]
        object_text += f"A {c} {s} object in {t} is visible. "
    
    return [
                {
                    "role": "system",
                    "content": object_text
                }
            ], LIGHT_DICT_TEXT[label]


# Select translation functions
if symbolic:
    get_instructions = get_instructions_symbolic
    get_example = get_example_symbolic
    get_trial = get_trial_symbolic
else:
    get_instructions = get_instructions_text
    get_example = get_example_text
    get_trial = get_trial_text



# Generate output
output_data = []
for sample in input_data:
    new_samples = []
    input_sample = [get_instructions()]
    for trial in sample:
        if trial["light_state"] != "no": # example
            input_sample += get_example(trial["objects"], trial["light_state"])
        else: # test case
            input_test = input_sample.copy()
            trial_input, trial_ideal = get_trial(trial["objects"], trial["label"])
            
            new_samples.append({
                "input": input_test + trial_input,
                "ideal": trial_ideal
            })
    output_data += new_samples

print(f"Generated {len(output_data)} samples.")


# Print to output file
with open(output_file_path, "w") as f:
    for sample in output_data:
        f.write(json.dumps(sample) + "\n")

print("Done.")