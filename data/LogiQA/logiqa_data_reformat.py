import json
import os

f = open("Train.txt",'r')

byt = f.readlines()
total_list = []
save_filename = "Train.json"
id = 0
for number in range(0, len(byt), 8):
    label = 0
    if byt[number+1].replace("\n","") == "a":
        label = 0
    elif byt[number+1].replace("\n","") == "b":
        label = 1
    elif byt[number+1].replace("\n","") == "c":
        label = 2
    elif byt[number+1].replace("\n","") == "d":
        label = 3
    dict = {"context": byt[number+2].replace("\n",""),
            "question":byt[number+3].replace("\n",""),
            "answers":[byt[number+4].replace("\n","").replace("A.",""), byt[number+5].replace("\n","").replace("B.",""), byt[number+6].replace("\n","").replace("C.",""), byt[number+7].replace("\n","").replace("D.","")],
            "label": label,
            "id_string": "train_"+str(id)}
    id = id + 1
    total_list.append(dict)

with open(save_filename, 'w') as f:
    json.dump(total_list, f)