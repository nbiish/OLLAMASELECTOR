import random 
import pandas as pd
import requests

class DataLoader_SimpleQA(object):
    def __init__(self,random_state=2024,
                 num_query = 100000, # number of questions
                 cate='synthetic_short',
                ):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.num_query = num_query
        self.df = pd.read_csv(
    "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        )

    def get_query_list(self,category = 'dev'):
        queryset = [self._convert(idx) for idx in range(self.num_query)]
        #queryset = queryset[0:5]
        return queryset

    def _convert(self,index):
        q = self.task_data['examples'][index]['input']
        a = self.task_data['examples'][index]['target']
        return q, a, "NA", str(index)

    def get_query_df(self,):
        new_df = pd.DataFrame({
    'query': self.df['problem'],
    'true_answer': self.df['answer'],
    'image': 'NA',  # Set 'image' column to 'NA' for all rows
    'ID':  range(len(self.df)) #
        })
        new_df = new_df.head(self.num_query)
        return new_df