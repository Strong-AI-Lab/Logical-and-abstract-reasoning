import json

js_lines = []
num_of_rules = 5
# open jsonl files as read only
with open("/data/qbao775/Logical-and-abstract-reasoning/data/PARARULE-Plus/Depth"+str(num_of_rules)+"/PARARULE_Plus_Depth"+str(num_of_rules)+"_shuffled_test.jsonl", "r", encoding="utf-8") as file:
    # read file line by line
    for line in file:
        # use json.loads to a dictionary
        js_lines.append(json.loads(line))
        
# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/PARARULE-Plus/Depth"+str(num_of_rules)+"/PARARULE_Plus_Depth"+str(num_of_rules)+"_shuffled_test_evals_no_instruction.jsonl", "w", encoding="utf-8") as output_file:
    for item in js_lines:
        for ques in item["questions"]:
            temp = {"input": [{"role": "system", "content": None},
                        {"role": "user", "content": None}], 
                "ideal": None}
            temp["input"][0]["content"] = "Passage: " + item["context"]
            temp["input"][1]["content"] = "Question: " + ques["text"]
            # label = None
            temp["ideal"] = ques["label"]
            json.dump(temp, output_file, ensure_ascii=False)
            output_file.write("\n")

        