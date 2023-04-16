# Logical and Abstract Reasoning

Repository for the evaluation of Large Language Models on logical and abstract reasoning tasks

## Installation

```
conda create -n logiarc python=3.8
conda activate logiarc
git clone https://github.com/Strong-AI-Lab/Logical-and-abstract-reasoning.git
cd Logical-and-abstract-reasoning
pip install -r requirements.txt
```

You also need to install apex, you can following the steps if you are using the Linux system.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Models


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
      <th rowspan="3" colspan="2" align="center" valign="middle">Logical Reasoning on Reading Comprehension</th>
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
      <th rowspan="9" colspan="2" align="center" valign="middle">Abstract Reasoning</th>
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
  <!--<tr>
      <td align="center">PGM</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://icml.cc/Conferences/2018/Schedule?showEvent=2194">paper</a> <br /> <a href="https://github.com/deepmind/abstract-reasoning-matrices">code</a>  </td>
      <td align="center">Text version of a Visual Abstract Reasoning task</td>
  </tr>
  <tr>
      <td align="center">RAVEN</td>
      <td align="center">-</td>
      <td align="center">Abstract Reasoning</td>
      <td align="center"> <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_RAVEN_A_Dataset_for_Relational_and_Analogical_Visual_REasoNing_CVPR_2019_paper.html">paper</a> <br /> <a href="http://wellyzhang.github.io/project/raven.html">project</a>  </td>
      <td align="center">Text version of a Visual Abstract Reasoning task</td>
  </tr>-->
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



