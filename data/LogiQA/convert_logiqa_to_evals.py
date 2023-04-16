import json


# read JSON file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA/Test.json", "r", encoding="utf-8") as input_file:
    data = json.load(input_file)

# save to a new JSONL file
with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA/Test_evals.jsonl", "w", encoding="utf-8") as output_file:
    for item in data:
        temp = {"input": [{"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails."},
                        {"role": "user", "content": None}], 
                "ideal": None,
                "id_string": None}
        temp["input"][1]["content"] = "\nPassage: " + item["context"] + " Question: " + item["question"] + " \nA. " + "A. " + item["answers"][0] + \
            " \nB. " + "B. " + item["answers"][1] + " \nC. " + "C. " + item["answers"][2] + " \nD. " + "D. " + item["answers"][3] + " \nAnswer: " 
        label = None
        if item["label"] == 0:
            label = "A"
        elif item["label"] == 1:
            label = "B"
        elif item["label"] == 2:
            label = "C"
        elif item["label"] == 3:
            label = "D"
        temp["ideal"] = label
        temp["id_string"] = item["id_string"]
        json.dump(temp, output_file, ensure_ascii=False)
        output_file.write("\n")
