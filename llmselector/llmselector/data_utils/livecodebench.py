import random 
import pandas as pd
from datasets import load_dataset

class DataLoader_livecodebench(object):
    def __init__(self,random_state=2024,category = 'test'):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.name_mapper = {"execution":"livecodebench/execution-v2"}

    def get_query_list(self,category = 'execution'):
        name_mapper = self.name_mapper
        dataset = ds = load_dataset("livecodebench/execution-v2")
        problems = dataset['test']
        queryset = [self._convert(a,idx) for idx,a in enumerate(problems)]
        return queryset
        
    def _convert(self,a, index):
        instruction = 'What is the output of the following code given the input? Think step by step and then give your final answer by "the answer is [xxx]."\n'
        query = f"Code: {a['code']}\nInput: {a['input']}\nOutput:"    
        answer = a['output']
        return instruction+query, answer, "NA", str(a['question_id'])+"__"+str(a['id'])

    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        return df