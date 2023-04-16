import json
import random

# read JSON file
js_lines = []
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test.txt", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        js_lines.append(json.loads(line))

# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test_none_answer_evals.jsonl", "w", encoding="utf-8") as output_file:
    for item in js_lines:
        temp = {"input": [{"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails."},
                        {"role": "user", "content": None}], 
                "ideal": None,
                "id_string": None,
                "type": None}
        option_list = [item["options"][0], item["options"][1], item["options"][2], item["options"][3]]
        label = None
        if option_list[0] == item["options"][item["answer"]]:
            label = "A"
        elif option_list[1] == item["options"][item["answer"]]:
            label = "B"
        elif option_list[2] == item["options"][item["answer"]]:
            label = "C"
        elif option_list[3] == item["options"][item["answer"]]:
            label = "D"
        option_list[item["answer"]] = "None of the other options are correct."
        temp["input"][1]["content"] = "\nPassage: " + item["text"] + " Question: " + item["question"] + " \nA. " + "A. " + option_list[0] + \
            " \nB. " + "B. " + option_list[1] + " \nC. " + "C. " + option_list[2] + " \nD. " + "D. " + option_list[3] + " \nAnswer: " 
        temp["ideal"] = label
        temp["id_string"] = item["id"]
        temp["type"] = item["type"]
        json.dump(temp, output_file, ensure_ascii=False)
        output_file.write("\n")
