import json

js_lines = []
num_of_rules = 5
tag = "test"
# open jsonl files as read only
with open("/data/qbao775/Logical-and-abstract-reasoning/data/PARARULE-Plus/Depth"+str(num_of_rules)+"/PARARULE_Plus_Depth"+str(num_of_rules)+"_shuffled_"+tag+".jsonl", "r", encoding="utf-8") as file:
    # read file line by line
    for line in file:
        # use json.loads to a dictionary
        js_lines.append(json.loads(line))
        
# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/PARARULE-Plus/Depth"+str(num_of_rules)+"/PARARULE_Plus_Depth"+str(num_of_rules)+"_shuffled_"+tag+"_huggingface.jsonl", "w", encoding="utf-8") as output_file:
    for item in js_lines:
        for ques in item["questions"]:
            temp = {"id": None, 
                    "context": None,
                    "question": None,
                    "label":None,
                    "meta":None}
            temp["id"] = ques["id"]
            temp["context"] = item["context"]
            temp["question"] = ques["text"]
            if ques["label"] == "true":
                temp["label"] = 1
            elif ques["label"] == "false":
                temp["label"] = 0
            temp["meta"] = ques["meta"]
            json.dump(temp, output_file, ensure_ascii=False)
            output_file.write("\n")

        