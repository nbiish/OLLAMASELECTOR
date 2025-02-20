import random 
import pandas as pd

class DataLoader_CommonGenHard(object):
    def __init__(self,path = 'data/commongen/commongen_hard.jsonl'):
        self.file_path = path
        return

    def get_query_list(self,):
        file_path = self.file_path

        # Read the JSONL file into a DataFrame
        data = pd.read_json(file_path, lines=True)
        formatted_list = [
    [f"Concept: {', '.join(item)}.\nTask: Generate a sentence that contains all concept words.", item, "NA", idx ] for idx, item in enumerate(data['concepts'])
        ]
        return formatted_list

    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        return df