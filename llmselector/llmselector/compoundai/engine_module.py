# Navie Backtracker
class BacktrackerFull:
    def track(self,
              query,
              history,
              HistoryFull):
        return 1

class CriticNaive:
    def judge(self,
              query,
                history,
                HistoryFull,
                            metric="",
              answer="",
                ):
        return 1

class CriticCount(CriticNaive):
    def __init__(self,count=10):
        self.count=count
    def judge(self,
              query,
              history,
              HistoryFull,
                            metric="",
              answer="",
                ):
        if(len(history)<3):
            return 1
        else:
            #print("length of 3")
            return get_score_count(history[-1][1],self.count)

class CriticOracle(CriticNaive):
    def __init__(self,count=10):
        self.count=count
    def judge(self,
              query,
              history,
              HistoryFull,
              metric="",
              answer="",
                ):
        if(len(history)<3):
            return 1
        else:
            #print("length of 3")
            return metric.get_score(history[-1][1],answer)

class CriticCascade(CriticNaive):
    def __init__(self,thres=9,idx_max=1):
        self.thres=thres
        self.idx_max=idx_max
    def judge(self,
              query,
              history,
              HistoryFull,
              metric="",
              answer="",
                ):
        idx = len(HistoryFull)
        #print(idx)
        if(idx>=self.idx_max):
            #print("hit max, always 1")
            return 1
        #print(history[-1])
        if('finish' not in history[-1][1]):
            return 1
        #print("has finish")
        iter = history[-1][1]['iter']
        #print(f"--iter is {iter}")
        finish = history[-1][1]['finish']
        if(finish==0 and iter<self.thres):
            #print("not finish return")
            return 1
        iter = history[-1][1]['iter']
        #print(f"--iter is {iter}")
        if(iter>=self.thres):
            
            #print(f"iter is too large goes to next model for idx {idx}")
            return 0
        return 1

class AllocatorNaive:
    def __init__(self,model="Claude-3-5-sonnet-20240620"):
        self.model = model
    def get_model(self,
                  query,
                 history,
                 HistoryFull,
             ):
        return self.model
        return 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo' 
        return "gpt-4o-2024-05-13"
        return "claude-3-5-sonnet-20240620"

    
class AllocatorFixChain(AllocatorNaive):
    def __init__(self,):
        pass
    
    def get_model(self,
                  query,
                 history,
                 HistoryFull,
             ):
        index = len(history)
        return self.model_list[index]

    def setup(self,
              model_list = [],
             ):
        self.model_list = model_list

class AllocatorSearch(AllocatorNaive):
    def setup(self,
              model_list = [],
              T=3,
             ):
        self.model_list = model_list
        self.T = T
        
    def get_model(self,
                  query,
                 history,
                 HistoryFull,
             ):
        return "claude-3-5-sonnet-20240620"


class AllocatorSearch(AllocatorNaive):
    def setup(self, model_list=[], T=2,
             same_model=False,
             ):
        self.model_list = model_list
        self.T = T
        self.same_model=same_model
        
    def get_model(self, query, history, HistoryFull):
        #return "claude-3-5-sonnet-20240620"
        
        # Create all possible products of models of length T
        all_combinations = list(product(self.model_list, repeat=self.T))
        if(self.same_model):
            all_combinations = [[item] * self.T for item in self.model_list]
        #print(f"all_combinations_length {len(all_combinations)}")
        
        # Get the used products by extracting the first elements from each inner list in HistoryFull
        used_products = {tuple(sublist[0] for sublist in product_pair) for product_pair in HistoryFull}
        #print(f"all {all_combinations}")
        #print(f"used_products {used_products}")
        
        # Generate prefix from history
        prefix_length = len(history)
        h0 = [a[0] for a in history]
        # Filter combinations to find valid ones
        valid_combinations = [
            combo for combo in all_combinations 
            if combo[:prefix_length] == tuple(h0) and tuple(combo) not in used_products
        ]
        #print(f"the current history {tuple(h0)}")
        #print(f"prefix_length {prefix_length}")
        #print("valid_combinations",valid_combinations)
        if valid_combinations:
            # Sample one from valid combinations
            #c = random.choice(valid_combinations)
            c = valid_combinations[0]
            #print(f"returned{c[prefix_length+1]}")
            return c[prefix_length]
        else:
            # Return a random model from model_list if no valid combination is found
            #print(f"used_products {len(used_products)}")
            #print(f"the current history {tuple(h0)}")

            c = self.model_list[0]
            #random.choice(self.model_list)
            #print(f"returned random {c} for {h0}")
            #print(f"---\n\n\n\n{query}\n\n\n---\n\n\n{history}\n\n\---\n\n\n{HistoryFull}---\n\n\n\n")
            return c

class AllocatorCascade(AllocatorNaive):
    def __init__(self, 
                 model_list=[
                     'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                     'claude-3-5-sonnet-20240620',
                 ],
                 
                ):
        self.model_list = model_list
        
    def setup(self, 
              model_list=[],
             ):
        self.model_list = model_list
        self.T = T
        self.same_model=same_model
        
    def get_model(self, query, history, HistoryFull):
        idx = len(HistoryFull)
        '''
        if(idx!=0):
            print("allocator idx {idx}")
        '''
        return self.model_list[idx]