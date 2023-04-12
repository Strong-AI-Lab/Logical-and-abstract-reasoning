## PARARULE-Plus Data Preprocessing
### Convert PARARULE-Plus data to openai/evals jsonl format with instruction
You can use the following script to run the data preprocessing and the generated files are saved as `PARARULE_Plus_Depth2_shuffled_train_evals.jsonl`, `PARARULE_Plus_Depth2_shuffled_dev_evals.jsonl`, and `PARARULE_Plus_Depth2_shuffled_test_evals.jsonl`.
~~~bash
python convert_pararule_plus_to_evals.py
~~~