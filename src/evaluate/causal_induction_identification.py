
import argparse
import json
import re

from evaluator import Evaluator


# Text translation functions
LIGHT_DICT_SYMB = {"off": 0, "undetermined": 1, "on": 2}
LIGHT_DICT_TEXT = ["off", "undetermined", "on"]
with open("../../data/ACRE/object_dict.json", "r") as object_dict_file:
    OBJECT_DICT_TEXT = json.load(object_dict_file)

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
                    "role": "user",
                    "content": f"{str(objects)} -> "
                }
            ], label
def parse_input_symbolic(s, nb_remove=4):
    s = re.sub(r"\{(.*?)\}, ", "", s, count=nb_remove)
    return s
def parse_label_symbolic(s):
    return re.sub(r"tensor\((.*?)\)", r"\1", s)

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
        object_text += f"A {c} {s} in {t} is visible. "
    object_text += "The state of the light is "
    
    return [
                {
                    "role": "user",
                    "content": object_text
                }
            ], LIGHT_DICT_TEXT[label]
def parse_input_text(s, nb_remove=4):
    # s = re.findall(r"\{(.*?)\}", s)[-1]
    s = re.sub(r"\{(.*?)\}, ", "", s, count=nb_remove)
    s = re.sub(r" object", "", s)
    s = re.sub(r"The light is A", "A", s) # needed to correct bug in old logs
    s = re.sub(r"What is the state of the light\?A", "A", s) # needed to correct bug in old logs
    s = re.sub(r"What is the state of the light\?'", r"The state of the light is '", s) # needed to updated format of old logs
    s = re.sub(r"The light is '", r"The state of the light is '", s)
    # s = "[{" + s + "}]"
    return s
def parse_label_text(s):
    if isinstance(s, int) or s.isnumeric():
        s = LIGHT_DICT_TEXT[int(s)]
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str, help='Path to the file containing the results to identify.')
    parser.add_argument('truth_file', type=str, help='Path to the file containing the causal induction ground truth.')
    parser.add_argument('--symbolic', action='store_true', help='Whether to use symbolic format.')
    parser.add_argument('--cot', action='store_true', help='Whether to use COT format.')

    args = parser.parse_args()

    result_file = args.result_file
    truth_file = args.truth_file
    symbolic = args.symbolic

    if args.cot:
        nb_input_remove = 1
    else:
        nb_input_remove = 4

    # Select translation functions
    if symbolic:
        get_example = get_example_symbolic
        get_trial = get_trial_symbolic
        parse_input = lambda s : parse_input_symbolic(s,nb_input_remove)
        parse_label = parse_label_symbolic
    else:
        get_example = get_example_text
        get_trial = get_trial_text
        parse_input = lambda s : parse_input_text(s,nb_input_remove)
        parse_label = parse_label_text

    # Read raw data
    with open(truth_file) as f:
        input_data = json.load(f)
        
    match_keys = {}
    for sample in input_data:
        new_samples = []
        trial_input = []
        for trial in sample:
            if "type" not in trial: # example
                trial_input += get_example(trial["objects"], trial["light_state"])
            else: # test case
                type = trial["type"]
                trial_query, trial_ideal = get_trial(trial["objects"], trial["label"])
                match_keys[(str(trial_input + trial_query), str(trial_ideal))] = type
        
    # Collect results
    evaluator = Evaluator(result_file,
                        arrow=True,
                        # num=True,
                        # cot=True,
                        # keywords_cot=True,
                        select_ans="first",
                        # pos_tagging=True,
                        # answer_type="num",
                        )
    
    rows = evaluator.get_results().values.tolist()
    mean_acc, accs = evaluator.get_accuracy()

    print("Mean accuracy: {}".format(mean_acc))

    causal_type_accs = {
        "direct": [],
        "indirect": [],
        "screen_off": [],
        "potential": [],
    }
    for i, acc in enumerate(accs):
        str_row_input = parse_input(rows[i][-4])
        str_row_label = parse_label(rows[i][-2])

        try:
            type = match_keys[(str_row_input, str_row_label)]
        except KeyError:
            reason = ""
            if str_row_input not in [key[0] for key in match_keys.keys()]:
                reason += "Input not found in keys. "
            if str_row_label not in [key[1] for key in match_keys.keys()]:
                reason += "Label not found in keys. "
            if str_row_label not in [key_1 for key_0, key_1 in match_keys.keys() if key_0 == str_row_input]:
                reason += f"Label not found for this input. Options: ({', '.join([key_1 for key_0, key_1 in match_keys.keys() if key_0 == str_row_input])}). "
            if reason == "":
                reason = "Unknown."
            raise KeyError(f"No match found for this input and label: ({str_row_input}, {str_row_label}). {reason}")

        causal_type_accs[type].append(acc)

    print("Direct accuracy: {} ({}/{})".format(sum(causal_type_accs["direct"]) / len(causal_type_accs["direct"]), sum(causal_type_accs["direct"]), len(causal_type_accs["direct"])))
    print("Indirect accuracy: {} ({}/{})".format(sum(causal_type_accs["indirect"]) / len(causal_type_accs["indirect"]), sum(causal_type_accs["indirect"]), len(causal_type_accs["indirect"])))
    print("Backward-blocking accuracy: {} ({}/{})".format(sum(causal_type_accs["potential"]) / len(causal_type_accs["potential"]), sum(causal_type_accs["potential"]), len(causal_type_accs["potential"])))
    print("Screen-off accuracy: {} ({}/{})".format(sum(causal_type_accs["screen_off"]) / len(causal_type_accs["screen_off"]), sum(causal_type_accs["screen_off"]), len(causal_type_accs["screen_off"])))


        