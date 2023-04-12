import json


# read JSON file
js_lines = []
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test.txt", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        js_lines.append(json.loads(line))

# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test_evals.jsonl", "w", encoding="utf-8") as output_file:
    for js in js_lines:
        temp = {"input": [{"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails."},
                        {"role": "user", "content": None}], 
                "ideal": None,
                "id_string": None,
                "type": None}
        temp["input"][1]["content"] = "\nPassage: " + js["text"] + " Question: " + js["question"] + " \nA. " + "A. " + js["options"][0] + \
            " \nB. " + "B. " + js["options"][1] + " \nC. " + "C. " + js["options"][2] + " \nD. " + "D. " + js["options"][3] + " \nAnswer: " 
        label = None
        if js["answer"] == 0:
            label = "A"
        elif js["answer"] == 1:
            label = "B"
        elif js["answer"] == 2:
            label = "C"
        elif js["answer"] == 3:
            label = "D"
        temp["ideal"] = label
        temp["id_string"] = js["id"]
        temp["type"] = js["type"]
        json.dump(temp, output_file, ensure_ascii=False)
        output_file.write("\n")
