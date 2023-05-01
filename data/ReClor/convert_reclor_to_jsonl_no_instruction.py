import json

# read JSON file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/ReClor/test.json", "r", encoding="utf-8") as input_file:
    data = json.load(input_file)
flag = "test"
# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/ReClor/ReClor_test_no_instruction.jsonl", "w", encoding="utf-8") as output_file:
    for item in data:
        temp = {"input": [{"role": "system", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None},
                        {"role": "user", "content": None}], 
                "choice_strings": ["A", "B", "C", "D"],
                "ideal": None}
        temp["input"][0]["content"] = "Passage: " + item["context"]
        temp["input"][1]["content"] = "Question: " + item["question"]
        temp["input"][2]["content"] = "A. " + item["answers"][0]
        temp["input"][3]["content"] = "B. " + item["answers"][1]
        temp["input"][4]["content"] = "C. " + item["answers"][2]
        temp["input"][5]["content"] = "D. " + item["answers"][3]
        label = None
        if flag != "test":
            if item["label"] == 0:
                label = "A"
            elif item["label"] == 1:
                label = "B"
            elif item["label"] == 2:
                label = "C"
            elif item["label"] == 3:
                label = "D"
        temp["ideal"] = label
        json.dump(temp, output_file, ensure_ascii=False)
        output_file.write("\n")
