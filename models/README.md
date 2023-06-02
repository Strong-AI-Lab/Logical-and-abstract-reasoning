# Models section
## ChatGPT prompt tuning script
You can run the following command to conduct the experiment for call chatgpt api to do the test. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`. Here is an example to use ChatGPT to do testing. You can use your own openai api key to replace the `OPENAI_API_KEY` in `openai.api_key = "OPENAI_API_KEY"`.
```
python run_chatgpt_prompt.py
```

You can follow this [web link](https://help.socialintents.com/article/188-how-to-find-your-openai-api-key-for-chatgpt) to create and use your own openai api key.

## Alpaca prompt tuning script
You can run the following command to conduct the experiment on Alpaca to do the test. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`. Here is an example to use alpaca to do testing.
```
python run_alpaca_prompt.py
```

## Convert model to support transformers
Currently `alpaca-7B` are trained and located at `/data/qbao775/Explanation-Generation/qiming_alpaca_7B` under `hgx2.sail.cloud.nesi.nz` server. If you want to use Alpaca do conduct experiment, you may need to use `hgx2.sail.cloud.nesi.nz` server to do the experiment. `LLaMA` is located in `/data/LLaMA` under both `hgx1.sail.cloud.nesi.nz` and `hgx2.sail.cloud.nesi.nz` server. But if you want to load the `LLaMA` to do fine-tuning and testing, you need to convert the model into the format into huggingface version as the following shown.

```
## Convert the LLaMA-7B to LLaMA-7B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/7B \
    --model_size 7B \
    --output_dir llama_7B_hf
```

## LLaMA-based model instruction fine-tunning
If you want to fine-tune LLaMA, please follow the following script. Here is an example to replicate Alpaca using LLaMA. We refer the code from [here](https://github.com/tatsu-lab/stanford_alpaca). You need to git clone the [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca.git) firstly as the following. The dataset `alpaca_data.json` is also been provided under the [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca.git) project.

```
git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca
```

And then conduct the follwoing fine-tuning. Cause our requirements have included the packages that been used in stanford_alpaca, so you do not need to run the `pip install -r requirements.txt` again.

```
## Fine-tuning the LLaMA-7B and replicate the Alpaca model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./alpaca_data.json \
   --bf16 True \
   --output_dir qiming_alpaca \
   --num_train_epochs 3 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True
```

If you met some problems when you fine-tune LLaMA, you can have a look at this [link](https://github.com/tatsu-lab/stanford_alpaca/issues/159#issuecomment-1490247999) which I summarized some of my experience.

## Other encoder-based large language models from ReClor leaderboard training and evaluating scripts
You can run the following command to conduct the fine-tuning experiment on encoder-based large language models ([MERIt](https://github.com/SparkJiao/MERIt), [AMR-LE](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition) and [LReasoner](https://github.com/WangsyGit/LReasoner)) to do the training. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`. Here is an example do conduct fine-tuning using `ReClor` dataset. 
```
sh ./scripts/run_LLM_reclor.sh
```
You can also change the argument for only conducting test like removing the following arguments and only keep the `--do_test`.
```
--do_train \
--evaluate_during_training \
```

## fine-tuning/evaluation scripts for NLI and PARARULE-Plus tasks
You can run the following command to conduct the experiment on Alpaca to do the test. The code includes datasets from `GLUE` including `MNLI`, `MRPC`, `QNLI`, `RTE`, `QQP`, `PARARULE-Plus-Depth-2`, `PARARULE-Plus-Depth-3`, `PARARULE-Plus-Depth-4`, and `PARARULE-Plus-Depth-5`.
```
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name qbao775/PARARULE-Plus-Depth-2\
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/PARARULE-Plus-Depth-2/
```
