# 🎓 LLMSELECTOR: Optimizing Model Selection for Compound AI Systems


LLMSELECTOR is a framework designed for optimizing  _model selection of compound AI systems_.

## 🚀 What is LLMSELECTOR?

Researchers and developers are increasingly invoking multiple LLM calls in a compound AI system to solve complex tasks. But which LLM should be selected for each call? LLMSELECTOR is a framework that simplifies and optimizes model selection for compound systems seamlessly. In particular, it offers a programming model to build compound AI systems involving multiple LLM calls, and an optimized engine to select which models to use for different modules in these systems. We provide a few examples to show how to use LLMSELECTOR. 

## 💻 How to Use LLESELECTOR?
#### 🔧 Installation
You can install LLMSELECTOR by running the following commands:

```
git clone https://github.com/LLMSELECTOR/LLMSELECTOR
cd LLMSELECTOR
pip install -e ./llmselector
```
 
#### 💡 Quickstart (No API key needed)

To start, let us first set up the environment.

```python
import llmselector
if not os.path.exists('../cache/db_livecodebench.sqlite'): 
    !wget -P ../cache https://github.com/LLMSELECTOR/LLMSELECTOR/releases/download/0.0.1/db_livecodebench.sqlite
llmselector.config.config(
    db_path=f"../cache/db_livecodebench.sqlite" )
```

Next, let us load the livecodebench dataset.

```python
from llmselector.data_utils.livecodebench import DataLoader_livecodebench 
from sklearn.model_selection import train_test_split
Mydataloader = DataLoader_livecodebench()
q_data = Mydataloader.get_query_df()
train_df, test_df = train_test_split(q_data,test_size=0.5, random_state=2025)
```

Let us first evaluate self-refine systems using fixed models.

```python
from llmselector.compoundai.optimizer import OptimizerFullSearch
from llmselector.compoundai.metric import Metric, compute_score
model_list = ['gpt-4o-2024-05-13','claude-3-5-sonnet-20240620','gemini-1.5-pro']
Agents_SameModel ={}
for name in model_list:
    Agents_SameModel[name] = SelfRefine()
    Opt0 = OptimizerFullSearch(model_list = [name])
    Opt0.optimize( train_df, Metric('em'), Agents_SameModel[name])
results = compute_score(Agents_SameModel, test_df, Metric('em'))
print(results)
```
The expected output is 

| Name                     | Mean_Score |
|--------------------------|------------|
| gpt-4o-2024-05-13        | 0.862500   |
| claude-3-5-sonnet-20240620 | 0.891667   |
| gemini-1.5-pro           | 0.866667   |


Now, let us use LLMSELECTOR to optimize the system.

```python
from llmselector.compoundai.optimizer import OptimizerLLMDiagnoser
LLMSELECTOR = SelfRefine()
Optimizer = OptimizerLLMDiagnoser()
Optimizer.optimize( train_df, Metric('em'), LLMSELECTOR)
results = compute_score({"LLMSELECTOR":LLMSELECTOR}, test_df, Metric('em'))
print(results)
```
The expected output should be

| Name                     | Mean_Score |
|--------------------------|------------|
|  LLMSELECTOR             |   0.954167 |

I.e., LLMSELECTOR offers a notable performance gain (6%) compared to always using any fixed model.

#### 📖 More examples (No API key needed)

More examples can be found in ```examples/```.

#### 🌐 Customized systems and tasks (API keys needed)

To use LLMSELECTOR for your own compound AI systems and tasks, it is as easy as creating the systems and tasks and then invoking LLMSELECTOR.

- Create your system: create the components and pipelines similar to SelfRefine defiend in ```compoundai/module/selfrefine```

- Create your task: create a DataLoader object similar to these in ```data_utils```


- Invoke LLMSELECTOR: You can simply use LLMSELECTOR by 
```Optimizer.optimize(train_df,Metric('em'),your_compound_system)```

Note that you will need to set up API keys for your own systems. To do so, you can simply use 

```
llmselector.config.config(
	db_path=f"cache.sqlite" ,
	openai_api_key="YOUR_OPENAI_KEY",
	anthropic_api_key="YOUR_ANTHROPIC_KEY",
	together_ai_api_key="YOUR_TOGETHERAI_KEY",
	gemini_api_key="YOUR_GEMINI_KEY")
```
    
## ✨ Can I request features and contribute?

Yes! We are happy to hear from you. Please feel free to open an issue for any feature request.

If you are interested in contributing, we would also be happy to coordinate on ongoing efforts! Please send an email to Lingjiao (lingjiao [at] stanford [dot] edu) 


## 📣 Updates & Changelog


### 🔹 2025.02.27 - The project is alive now!

  - ✅ Added the codebase, relevant examples, and demos

    
## 🎯 Reference

If you find LLMSELECTOR useful, we would appreciate if you can please cite our work as follows:


```
@article{chen2025llmselector,
  title={Optimizing Model Selection for Compound AI Systems},
  author={Chen, Lingjiao and Davis, Jared and Hanin, Boris and Bailis, Peter and Zaharia, Matei and Zou, James and Stoica, Ion},
  journal={arXiv preprint},
  year={2025}
}
```
