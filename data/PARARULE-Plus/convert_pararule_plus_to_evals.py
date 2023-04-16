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
with open("/data/qbao775/Logical-and-abstract-reasoning/data/PARARULE-Plus/Depth"+str(num_of_rules)+"/PARARULE_Plus_Depth"+str(num_of_rules)+"_shuffled_test_evals.jsonl", "w", encoding="utf-8") as output_file:
    for item in js_lines:
        for ques in item["questions"]:
            temp = {"input": [{"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. You need to answer true or false to the question. Read the question thoroughly and answer true or false. Read the passage thoroughly to ensure you know what the passage entails and you need to use "+str(num_of_rules)+" rules to answer the question."},
                            {"role": "user", "content": None}], 
                    "ideal": None,
                    "id_string": None}
            temp["input"][1]["content"] = "\nPassage: " + item["context"] + " Question: " + ques["text"] + " \nAnswer: " 
            temp["ideal"] = ques["label"]
            temp["id_string"] = ques["id"]
            json.dump(temp, output_file, ensure_ascii=False)
            output_file.write("\n")

        