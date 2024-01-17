# Logical and Abstract Reasoning

Repository for the evaluation of Large Language Models on logical and abstract reasoning tasks

## Installation

To install the repository, use the following command:

```
git clone https://github.com/Strong-AI-Lab/Logical-and-abstract-reasoning.git
```

To install the dependencies in a virtual environment, use the following:
```
cd Logical-and-abstract-reasoning
python -m venv env/
source env/bin/activate
pip install -r requirements.txt
```

You may need to install [transformers](https://huggingface.co/docs/transformers/index) from the repository:
```
pip install git+https://github.com/huggingface/transformers
```


## Use

### Evaluation

To evaluate a model in the repository, use the following command:
```
python run_evaluation config/model/<model_config.yaml> config/data/<data_config.yaml> --<kwarg_name> <kwarg>
```

You can choose the model to evaluate by changing the `<model_config.yaml>` file, and the dataset to evaluate the model on by changing the `<data_config.yaml>` file. You can add any additional arguments as `<kwargs>` (e.g. private API key for GPT models). 

By default, all the results are saved in a csv file in the `logs/` folder. You can re-compute the metrics from the evaluation run from this file by running the following:
```
python src/evaluate/evaluator.py logs/<results_file.csv>
```

### Fine-tuning

To fine-tune a model on a given dataset, run the following:
```
python run_finetuning.py config/model/<model_config.yaml> config/data/<data_config.yaml> config/trainer/<trainer_config.yaml>
```
The configuration files work similarly as for evaluation. The `<model_config.yaml>` file contains additoinal configuration for training. The logs are saved in `fine-tuning-output/` and the model weights are saved in `fine-tuning-saves/`.

Currently, only HuggingFace models can be fine-tuned.

#### LLaMA-based model instruction fine-tuning
We use the LLaMA-based model fine-tuning from the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) training script. If you want to conduct a LLaMA-based model on instruction fine-tuning, you can do that by following [this link](https://github.com/Strong-AI-Lab/Logical-and-abstract-reasoning/blob/main/models/README.md#llama-based-model-instruction-fine-tunning). 

## Models
<table>
  <tr>
      <th colspan="2" align="center">Inference Type</th>
      <th align="center">Model</th>
      <th align="center">Size</th>
      <th align="center">Task</th>
      <th align="center">Link</th>
      <th align="center">Remark</th>
  </tr >
  
  <tr>
      <th rowspan="10" colspan="2" align="center" valign="middle">Logical Reasoning on Reading Comprehension</th>
      <td align="center">MERIt</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://arxiv.org/abs/2203.00357">paper</a> <br /> <a href="https://github.com/SparkJiao/MERIt">project</a>  </td>
      <td align="center">#3 on the ReClor leaderboard</td>
  </tr>
  <tr>
      <td align="center">LReasoner</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://aclanthology.org/2022.findings-acl.127/">paper</a> <br /> <a href="https://github.com/WangsyGit/LReasoner">project</a>  </td>
      <td align="center">#6 on the ReClor leaderboard</td>
  </tr>
  <tr>
      <td align="center">AMR-LE</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition">project</a>  </td>
      <td align="center">#2 and #5 on the ReClor leaderboard</td>
  </tr>
  
  <tr>
      <td align="center">LLaMA</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://arxiv.org/abs/2302.13971">paper</a> <br /> <a href="https://github.com/facebookresearch/llama">code</a>  </td>
      <td align="center">Open source very large language model</td>
  </tr>
  <tr>
      <td align="center">LLaMA2</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://arxiv.org/abs/2307.09288">paper</a> <br /> <a href="https://huggingface.co/docs/transformers/main/model_doc/llama2">code</a>  </td>
      <td align="center">Open source very large language model</td>
  </tr>
  <tr>
      <td align="center">Alpaca</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://github.com/tatsu-lab/stanford_alpaca">code</a>  </td>
      <td align="center">Fine-tuned LLaMA</td>
  </tr>
  <tr>
      <td align="center">Vicuna</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://vicuna.lmsys.org/">project</a> </br> <a href="https://github.com/lm-sys/FastChat">code</a></td>
      <td align="center">Fine-tuned LLaMA</td>
  </tr>
  <tr>
      <td align="center">ChatGPT</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://arxiv.org/abs/2005.14165">paper</a> <br/><a href="https://openai.com/blog/chatgpt">project</a> </td>
      <td align="center">Use api to do prompt tuning</td>
  </tr>
  <tr>
      <td align="center">GPT-4</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://arxiv.org/abs/2303.08774">paper</a> <br/><a href="https://openai.com/product/gpt-4">project</a> </td>
      <td align="center">Use api to do prompt tuning</td>
  </tr>
  <tr>
      <td align="center">Zephyr-7b-beta</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta">code</a> </td>
      <td align="center">Fine-tuned Mistral-7b</td>
  </tr>
</table>

## Datasets & Benchmarks

<table>
  <tr>
      <th colspan="2" align="center">Inference Type</th>
      <th align="center">Dataset</th>
      <th align="center">Size</th>
      <th align="center">Task</th>
      <th align="center">Link</th>
      <th align="center">Remark</th>
  </tr >
  
  <tr>
      <th rowspan="4" colspan="2" align="center" valign="middle">Logical Reasoning on Reading Comprehension</th>
      <td align="center">ReClor</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://openreview.net/pdf?id=HJgJtT4tvB">paper</a> <br /> <a href="https://whyu.me/reclor/">project</a>  </td>
      <td align="center">Logical reasoning reading comprehension</td>
  </tr>
  <tr>
      <td align="center">LogiQA</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://www.ijcai.org/proceedings/2020/0501.pdf">paper</a> <br /> <a href="https://github.com/lgw863/LogiQA-dataset">project</a>  </td>
      <td align="center">Logical reasoning reading comprehension</td>
  </tr>
  <tr>
      <td align="center">LogiQA V2</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://github.com/openai/evals/pull/470">project</a>  </td>
      <td align="center">Logical reasoning reading comprehension</td>
  </tr>
  <tr>
      <td align="center">LogiQA Logical Reasoning Plus</td>
      <td align="center">-</td>
      <td align="center">Reading Comprehension</td>
      <td align="center"> <a href="https://github.com/openai/evals/pull/648">project</a>  </td>
      <td align="center">Logical reasoning reading comprehension for out-of-distribution evaluation</td>
  </tr>
  
  <tr>
      <th rowspan="10" colspan="2" align="center" valign="middle">Abstract Reasoning</th>
      <td align="center">ARC</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://arxiv.org/abs/1911.01547">paper</a> <br /> <a href="https://github.com/fchollet/ARC">code</a>  </td>
      <td align="center">Text version of a Visual Abstract Reasoning task</td>
  </tr>
  <tr>
      <td align="center">ACRE</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="http://arxiv.org/abs/2103.14232">paper</a> <br /> <a href="https://github.com/WellyZhang/ACRE">code</a>  </td>
      <td align="center">Text version of a Visual Abstract Reasoning task</td>
  </tr>
  <tr>
      <td align="center">PVR</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="http://arxiv.org/abs/2107.12580">paper</a> </td>
      <td align="center">Abstract Reasoning task</td>
  </tr>
  <tr>
      <td align="center">RAVEN</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_RAVEN_A_Dataset_for_Relational_and_Analogical_Visual_REasoNing_CVPR_2019_paper.html">paper</a> <br /> <a href="http://wellyzhang.github.io/project/raven.html">project</a>  </td>
      <td align="center">Text version of a Visual Abstract Reasoning task</td>
  </tr>
  <tr>
      <td align="center">Diagrammatic Logic</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/openai/evals/tree/main/evals/registry/data/diagrammatic_logic">code</a> </td>
      <td align="center">Extracted from OpenAI Evals</td>
  </tr>
  <tr>
      <td align="center">Logic</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/openai/evals/tree/main/evals/registry/data/logic">code</a> </td>
      <td align="center">Extracted from OpenAI Evals</td>
  </tr>
  <tr>
      <td align="center">Logic Statements</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/openai/evals/tree/main/evals/registry/data/logic-statements">code</a> </td>
      <td align="center">Extracted from OpenAI Evals</td>
  </tr>
  <tr>
      <td align="center">Pattern Identification</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/openai/evals/tree/main/evals/registry/data/pattern_identification">code</a> </td>
      <td align="center">Extracted from OpenAI Evals</td>
  </tr>
  <tr>
      <td align="center">String Patterns</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/openai/evals/tree/b592da66b33c103da42b6a6c8da40d8a3ea268d3/evals/registry/data/string_patterns">code</a> </td>
      <td align="center">Extracted from OpenAI Evals</td>
  </tr>
  <tr>
      <td align="center">List Functions</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/list_functions">code</a> </td>
      <td align="center">Extracted from Google BIG-bench</td>
  </tr>
  
  <tr>
      <th rowspan="1" colspan="2" align="center" valign="middle">Visual Abstract Reasoning</th>
      <td align="center"> ... </td>
      <td align="center"> </td>
      <td align="center"> </td>
      <td align="center"> </td>
      <td align="center"> </td>
  </tr>
</table>

## Acknowledgement
Our proposed new dataset [logiqa-logical-reasoning-plus](https://bit.ly/3MVjZNP) has been merged by [OpenAI/Evals](https://github.com/openai/evals).

