from itertools import product
from tqdm import tqdm
from .metric import get_score_count
import random, re, os, concurrent.futures
from sqlitedict import SqliteDict
# Make tqdm work with pandas apply
tqdm.pandas()
import pandas as pd
from .diagnoser import Diagnoser
from .engine_module import BacktrackerFull, AllocatorFixChain, CriticNaive

from collections import Counter
from functools import partial

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor


def mode_of_lists(list_of_lists):
    # Convert each inner list to a tuple so they can be used as keys in Counter
    tupled_lists = [tuple(lst) for lst in list_of_lists]
    
    # Use Counter to count occurrences of each tuple (representing a list)
    counter = Counter(tupled_lists)
    
    # Find the most common tuple (mode)
    mode_tuple, count = counter.most_common(1)[0]
    
    # Convert the tuple back to a list (or just return the tuple if needed)
    return list(mode_tuple)


class Optimizer(object):
    def __init__(self, 
                 model_list = ['gpt-4o-2024-05-13','claude-3-5-sonnet-20240620'],
                 max_budget=10000000,
                 parallel_eval=True,
                 max_worker=10,
                 backtracker = BacktrackerFull,
                 allocator = AllocatorFixChain,
                 critic = CriticNaive,
                 verbose = False,
                ):
        self.models = model_list
        self.max_budget=max_budget
        self.parallel_eval = parallel_eval
        self.max_worker=max_worker
        self.backtracker = backtracker
        self.allocator = allocator
        self.critic = critic
        self.verbose = verbose
        pass
        
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                ):
        pass
        
    def eval(self,
             data_df,
             AIAgent,
             M,
            ):
        # Function to apply the AIAgent.generate method
        def generate_answer(query):
            return AIAgent.generate(query)
        
        # Create a thread pool and process the queries in parallel
        def apply_parallel_with_progress(df, column, func, max_workers=20):
            queries = df[column].tolist()
            answers = []
        
            # Use ThreadPoolExecutor for multithreading
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks and wrap in tqdm for a progress bar
                futures = list(tqdm(executor.map(func, queries), total=len(queries), desc="Processing"))
                answers.extend(futures)
            return answers
            
        if((not(self.parallel_eval)) or len(data_df)==1):
            data_df['answer'] = data_df['query'].apply(AIAgent.generate)
        else:
            # Apply the function in parallel
            data_df['answer'] = apply_parallel_with_progress(data_df, 'query', generate_answer,self.max_worker)

        data_df['score'] = data_df.apply(lambda row: M.get_score(row['answer'], row['true_answer']), axis=1)
        return data_df['score'].mean()

    def set_budget(self,
                   max_budget=1000,
                  ):
        self.max_budget = max_budget
        return max_budget

    def set_models(self,
             combo,
             compoundaisystem,
             ):
        Myallow = self.allocator()
        Myallow.setup(combo)
        compoundaisystem.setup_ABC(allocator = Myallow, backtracker = self.backtracker(), critic = self.critic())
        return compoundaisystem

class OptimizerFullSearch(Optimizer):
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                ):
        def compute_score(combo):
            self.set_models(combo,compoundaisystem)
            score = self.eval(training_df,compoundaisystem,metric)
            return score
        # compute the score for all combinations
        T = len(compoundaisystem.get_pipeline())-1 # number of components
        all_permutations = list(product(self.models, repeat=T))
        random.shuffle(all_permutations)
        all_permutations = all_permutations[0:self.max_budget]
        scores = [(combo, compute_score(combo)) for combo in tqdm(all_permutations)]
        # get the maximum score and the config
        max_combo, max_score = max(scores, key=lambda x: x[1])
        # set up the model
        print(max_combo)
        compoundaisystem = self.set_models(max_combo,compoundaisystem)
        return compoundaisystem

class OptimizerManual(Optimizer):        
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                 model_best=['gpt-4o'],
                ):
        self.set_models(model_best,compoundaisystem)
        return
        
class OptimizerLLMDiagnoser(Optimizer):
    def __init__(self, 
                 model_list = [
              'gpt-4o-2024-05-13','gpt-4-turbo-2024-04-09','gpt-4o-mini-2024-07-18',
              'claude-3-5-sonnet-20240620','claude-3-haiku-20240307',
              'gemini-1.5-pro','gemini-1.5-flash',
              'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo','Qwen/Qwen2.5-72B-Instruct-Turbo',
              ],
                 max_budget=1000000,
                 parallel_eval=True,
                 max_worker=10,
                 backtracker = BacktrackerFull,
                 allocator = AllocatorFixChain,
                 critic = CriticNaive,
                 verbose = False,
                 
                 judge_model = 'claude-3-5-sonnet-20240620',
                 diag_model = 'gemini-1.5-pro',
                 seed = 0,
                 beta = 1,
                 alpha = 0,
                 get_answer=0,
                 max_workers=10,
                ):
        super().__init__(
                model_list = model_list,
                 max_budget=max_budget,
                 parallel_eval=parallel_eval,
                 max_worker=max_worker,
                 backtracker = backtracker,
                 allocator = allocator,
                 critic = critic,
                 verbose = verbose)
        self.judge_model = judge_model
        self.diag = Diagnoser(diag_model)
        random.seed(seed)
        self.beta = beta
        self.alpha = alpha
        self.get_answer = get_answer
        self.max_workers = max_workers
        pass

    def allocation2key(self,allocation):
        key = " ".join(allocation)
        return key

    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                 show_progress=False,
                 get_answer=0,
                 max_iter=1000,
                 init_mi=[],
                ):
        def check_last_elements_same(L, T):
            if T > len(L):  # If T is larger than the list, return False or handle appropriately
                return False
            last_elements = L[-T:]  # Slice to get the last T elements
            last_elements_as_tuples = [tuple(lst) for lst in last_elements]
            return len(set(last_elements_as_tuples)) == 1  # Check if all elements are the same using a set

        def aggregateallocation(allocations, allocation_list):
            mode = find_mode(allocations.to_list())
            #print(f"the most common allocation is {mode}") if self.verbose==True else None
            allocation_list.append(mode)
            return mode, allocation_list
     
        def find_mode(data):
            # Initialize a Counter to track the frequency of all possible values
            frequency = Counter()
        
            # Iterate over each sublist in the data
            for item in data:
                # Iterate over each possible value in the sublist
                for possible_value in item:
                    # Convert the value to a tuple to ensure hashability
                    frequency[tuple(possible_value)] += 1
        
            # Find the most common value (the mode)
            mode, count = frequency.most_common(1)[0]  # Gets the most common element and its count
            print(frequency) if self.verbose==True else None
            return list(mode)  # Convert the mode back into a list (if required)
        
        # initialization
        L = len(compoundaisystem.get_pipeline())-1
        c = 0
        M = len(self.models)-1
        B = self.max_budget
        iter = 0
        delta = 0
        allocator = list(tuple(random.choices(self.models, k=L)))
        if(len(init_mi)==L):
            allocator = init_mi
        allocator_list = [allocator]
        while(c<=B-M and delta == 0):
            print(allocator_list) if self.verbose else None
            # choose a module to optimize
            module_idx = self.update_module(iter=iter,L=L)
            # optimize the allocation w.r.t. the chosen module for each data point
            allocations = self.update_model(training_df, metric, compoundaisystem, allocator = allocator, module_idx = module_idx)
            # aggregate to one allocation
            allocator, allocator_list = aggregateallocation(allocations,allocator_list)
            print(f"----allocation list---- {allocator_list}") if self.verbose else None
            # check for stopping criteria
            c += M
            delta = check_last_elements_same(allocator_list,T=L)
            iter += 1
        self.set_models(allocator_list[-1],compoundaisystem)
        
        print("final allocation list:",allocator_list) if self.verbose else None
        return 

    def update_module(self, 
                   iter = 0,
                   module_idx = 0,
                   L = 3,
                  ):
        return (iter) % L
        
    def update_model(self,training_df, metric, compoundaisystem, allocator, module_idx = 0):
        def process_row(row):
            # compute the score for each possible model allocated to module_idx
            allocator_list = [allocator[:module_idx] + [model] + allocator[module_idx + 1:] for model in self.models]
            row_df = pd.DataFrame([row])
            scores = [(allocated_model,self.compute_score_onerow( row_df, metric,  compoundaisystem, allocated_model,module_idx)) for allocated_model in allocator_list]
            #print(scores) if self.verbose else None
            # Find the maximum score
            max_score = max(scores, key=lambda x: x[1])[1]
            # Filter all items with the maximum score
            max_combos = [combo for combo, score in scores if score == max_score]
            #print("all scoure max:",max_combos)
            # take the maximum
            max_combo, max_score = max(scores, key=lambda x: x[1])
            return max_combos

        def update_allocator(df, allocation):
            # Check if the 'allocator' column exists
            if 'allocator' not in df.columns:
                # Initialize with the allocation list
                df = df.copy()
                df.loc[:, 'allocator'] = [allocation] * len(df)

            else:
                # Update the 'allocator' column for each row
                def update_row(row):
                    if isinstance(row['allocator'], list) and len(row['allocator']) == 1:
                        # Use the first value to replace it
                        return [row['allocator'][0]]
                    else:
                        # Use allocation to replace it
                        return allocation
                
                df['allocator'] = df.apply(update_row, axis=1)
            
            return df
    
        training_df = update_allocator(training_df, allocator)

        '''
        # TODO: multi-thread requires a refactor
        # Currently it raises conflicts (due to handling compound ai systems' allocations)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for parallel execution to process each row
            futures = {idx: executor.submit(process_row, row) for idx, (_, row) in enumerate(training_df.iterrows())}
    
            # Collect results as they are completed
            results = [None] * len(training_df)  # Placeholder for results
            for future in as_completed(futures.values()):
                # Identify the completed future's index
                idx = next(idx for idx, fut in futures.items() if fut == future)
                results[idx] = future.result()
    
        # Step 3: Add the results to the DataFrame
        training_df['allocator'] = results
        '''
        
        training_df['allocator'] = training_df.progress_apply(process_row, axis=1)
        return training_df['allocator']

    def compute_score_onerow(self,
                             training_df, 
                             metric, 
                             compoundaisystem,
                             allocated_model,
                             module_idx,
                            ):
        self.set_models(allocated_model,compoundaisystem)
        score = self.eval(training_df, compoundaisystem, metric)  # Pass the row to eval, not the whole df
        if(self.alpha==0):
            return score*self.beta
        score_diagnoser = self.diagnose( training_df, metric, compoundaisystem, allocated_model,module_idx)
        return score*self.beta + self.alpha*score_diagnoser
        
    def diagnose(self, training_df, metric, compoundaisystem,  allocated_model,module_idx):
        error, analysis = self.get_score_LLM_onequery(training_df.iloc[0],compoundaisystem,module_idx=module_idx)
        #print(f"allocated model:: {allocated_model} and diag index {module_idx}") if self.verbose else None
        #print(analysis) if self.verbose else None
        return 1-error

    def get_score_LLM_onequery(self,
                    query_full,
                    compoundaisystem,
                    module_idx=0,
                     ):
        ans = compoundaisystem.generate(query_full['query'])
        history = compoundaisystem.load_history()['trace']
        Info1 = {'description':compoundaisystem.get_description()}   
        Info1['module']= [t[1] for t in history] 
        error, analysis = self.diag.diagnose(
                compoundaisystem=Info1,
                 query=query_full['query'],
                 answer=ans,
                 true_answer=query_full['true_answer'],
                 module_id = module_idx,
            temperature=0,
            show_prompt=False,
                    )
        return error, analysis
 