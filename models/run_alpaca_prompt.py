import pandas as pd
import json
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
# load_model_name = "./llama_7B_hf/llama-7b/"
load_model_name = "./qiming_alpaca_7B/"
flag = "reclor"

def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        #device_map="auto",
        torch_dtype=torch.float16,
        #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        #load_in_8bit=eight_bit,
        #from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

load_model(load_model_name)

    
response_list = {'context':[],'question':[],'optionA':[], 'optionB':[],'optionC':[],'optionD':[],'predict_answer':[]}
if flag == "reclor":
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/ReClor/test.json", "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    for item in tqdm(data):
        context = item["context"]
        question = item["question"]
        optionA = item["answers"][0]
        optionB = item["answers"][1]
        optionC = item["answers"][2]
        optionD = item["answers"][3]
        fulltext = "Instruction: Please only generate A, B, C or D as your predicted answer for the following input." + \
            " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        
        generated_text = ""
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
        in_tokens = len(gen_in)
        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

        response = text_without_prompt

        # response = response.split(human_invitation)[0]

        response = response.strip()
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response)
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_reclor_prediction.xlsx")
        
        
elif flag == "logiqa":
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA/Test.json", "r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    for item in tqdm(data):
        context = item["context"]
        question = item["question"]
        optionA = item["answers"][0]
        optionB = item["answers"][1]
        optionC = item["answers"][2]
        optionD = item["answers"][3]
        fulltext = "Instruction: Please only generate A, B, C or D as your predicted answer for the following input." + \
            " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        
        generated_text = ""
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
        in_tokens = len(gen_in)
        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

        response = text_without_prompt

        # response = response.split(human_invitation)[0]

        response = response.strip()
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response)
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_logiqa_prediction.xlsx")
        
elif flag == "logiqav2":
    data = []
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    for item in tqdm(data):
        context = item["text"]
        question = item["question"]
        optionA = item["options"][0]
        optionB = item["options"][1]
        optionC = item["options"][2]
        optionD = item["options"][3]
        fulltext = "Instruction: Please only generate A, B, C or D as your predicted answer for the following input." + \
            " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        
        generated_text = ""
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
        in_tokens = len(gen_in)
        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

        response = text_without_prompt

        # response = response.split(human_invitation)[0]

        response = response.strip()
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response)
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_logiqav2_prediction.xlsx")