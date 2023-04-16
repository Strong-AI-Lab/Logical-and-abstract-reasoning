# LogiQA
This dataset consists of 8,678 QA instances.(Train:7376; Eval:651; Test:651)

The files is divided into English version: Train.txt, Eval.txt, Test.txt, and Chinese version: zh_train.txt, zh_eval.txt, zh_test.txt. 

Each 8 lines constitute an example of a problem.  (8,678 * 8 = 69,424)

In each 8 lines: 

                 The first line is blank line;

                 The second is right choice;

                 The third is context;
 
                 The fourth is question;
    
                 The remaining four lines are four options.

### Convert LogiQA data to openai/evals jsonl format with instruction
You can use the following script to run the data preprocessing and the generated files are saved as `train_evals.jsonl`, `val_evals.jsonl`, and `test_evals.jsonl`.
~~~bash
python convert_logiqa_to_evals.py
~~~

### Convert LogiQA data to openai/evals jsonl format with instruction and shuffle the order of options
You can use the following script to run the data preprocessing and the generated files are saved as `train_shuffle_options_evals.jsonl` and `val_shuffle_options_evals.jsonl`.
~~~bash
python convert_logiqa_to_shuffle_evals.py
~~~

### Convert LogiQA data to openai/evals jsonl format with instruction and replace answer with "None of the other options are correct"
You can use the following script to run the data preprocessing and the generated files are saved as `train_none_answer_evals.jsonl` and `val_none_answer_evals.jsonl`.
~~~bash
python convert_logiqa_to_none_answer_evals.py
~~~

### Convert LogiQA data to openai/evals jsonl format with instruction, shuffle the order of options and replace answer with "None of the other options are correct"
You can use the following script to run the data preprocessing and the generated files are saved as `train_shuffle_none_answer_evals.jsonl` and `val_shuffle_none_answer_evals.jsonl`.
~~~bash
python convert_logiqa_to_shuffle_none_answer_evals.py
~~~