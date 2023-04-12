## ReClor Data Preprocessing
### Convert ReClor data to openai/evals jsonl format with instruction
You can use the following script to run the data preprocessing and the generated files are saved as `train_evals.jsonl`, `val_evals.jsonl`, and `test_evals.jsonl`.
~~~bash
python convert_reclor_to_evals.py
~~~

### Convert ReClor data to openai/evals jsonl format with instruction and shuffle the order of options
You can use the following script to run the data preprocessing and the generated files are saved as `train_shuffle_options_evals.jsonl` and `val_shuffle_options_evals.jsonl`.
~~~bash
python convert_reclor_to_shuffle_evals.py
~~~

### Convert ReClor data to openai/evals jsonl format with instruction and replace answer with "None of the other options are correct"
You can use the following script to run the data preprocessing and the generated files are saved as `train_none_answer_evals.jsonl` and `val_none_answer_evals.jsonl`.
~~~bash
python convert_reclor_to_none_answer_evals.py
~~~

### Convert ReClor data to openai/evals jsonl format with instruction, shuffle the order of options and replace answer with "None of the other options are correct"
You can use the following script to run the data preprocessing and the generated files are saved as `train_shuffle_none_answer_evals.jsonl` and `val_shuffle_none_answer_evals.jsonl`.
~~~bash
python convert_reclor_to_shuffle_none_answer_evals.py
~~~