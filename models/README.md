# Models section
## ChatGPT prompt tuning script
You can run the following command to conduct the experiment for call chatgpt api to do the test. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`.
```
python run_chatgpt_prompt.py
```

## Alpaca prompt tuning script
You can run the following command to conduct the experiment on Alpaca to do the test. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`.
```
python run_alpaca_prompt.py
```

## Other encoder-based large language models from ReClor leaderboard training and evaluating scripts
You can run the following command to conduct the fine-tuning experiment on encoder-based large language models (MERIt, AMR-LE and LReasoner) to do the training. The code includes datasets for `ReClor`, `LogiQA` and `LogiQA-V2`. Here is an example do conduct fine-tuning using `ReClor` dataset. 
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