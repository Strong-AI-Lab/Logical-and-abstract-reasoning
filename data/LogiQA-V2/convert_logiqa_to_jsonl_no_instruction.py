import json


# read JSON file
js_lines = []
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/dev.txt", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        js_lines.append(json.loads(line))

# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/LogiQA-V2_dev_no_instruction.jsonl", "w", encoding="utf-8") as output_file:
    for js in js_lines:
        temp = {"input": [{"role": "system", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None}], 
                "choice_strings": ["A", "B", "C", "D"],
                "ideal": None}
        temp["input"][0]["content"] = "Passage: " + js["text"]
        temp["input"][1]["content"] = "Question: " + js["question"]
        temp["input"][2]["content"] = "A. " + js["options"][0]
        temp["input"][3]["content"] = "B. " + js["options"][1]
        temp["input"][4]["content"] = "C. " + js["options"][2]
        temp["input"][5]["content"] = "D. " + js["options"][3]
        
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
        json.dump(temp, output_file, ensure_ascii=False)
        output_file.write("\n")
