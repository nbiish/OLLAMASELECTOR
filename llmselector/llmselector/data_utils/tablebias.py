import random 
import pandas as pd

class DataLoader_TableBias(object):
    def __init__(self,random_state=2024,
                 num_query = 1000, # number of questions
                 n=300,
                 int_max=100,
                 int_min=1,
                 id_length = 10,  # Length of the ID
                 category = 'dev',
                 random_seed = 2025,
                ):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.int_max = int_max
        self.int_min = int_min
        self.id_length = id_length
        self.random_seed = random_seed
        self.num_query = num_query
        self.n = n
        
    def get_query_list(self,category = 'dev'):
        queryset = [self._convert(idx) for idx in range(self.num_query)]
        #queryset = queryset[0:5]
        return queryset

    def _convert(self,index):
        text,id_dict, ans_dict, a_dict, b_dict = generate_table_multiply_questions(n = self.n,int_min=self.int_min,int_max=self.int_max,id_length=self.id_length,local_random=self.local_random)
        new_text, id_dict, q_id, q_question = convert_row_to_column(id_dict,local_random=self.local_random)
        return new_text, ans_dict[q_id], "NA", str(index), q_id, f"{a_dict[q_id]}"

    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID','task_id','formula'])
        return df

def generate_table_multiply_questions(n, int_min=1, int_max=100, id_length=10, local_random=random):
    """
    Generate a text with N lines, each containing a unique ID of specified length
    and a multiplication question with two K-digit integers. Also returns 
    a dictionary mapping the IDs to their respective questions.
    
    Args:
        n (int): Number of lines to generate.
        k (int): Number of digits for each integer a and b.
        id_length (int): Length of the unique ID (default is 10).
    
    Returns:
        tuple: A tuple containing:
            - A string: The generated text with N lines.
            - A dictionary: The dictionary with ID as key and question as value.
    """
    id_set = set()  # To keep track of unique IDs
    lines = []
    id_to_question = {}
    answer_dict = {}
    a_dict = {}
    b_dict = {}
    while len(lines) < n:
        random_id = ''.join(local_random.choices('0123456789', k=id_length))
        if random_id not in id_set:
            id_set.add(random_id)
            question, ans, x = generate_qa(local_random,int_max=int_max,int_min=int_min)
            lines.append(f"ID: {random_id}. Task: {question}")
            id_to_question[random_id] = question
            answer_dict[random_id] = ans
            a_dict[random_id] = x
            b_dict[random_id] = x
    return "\n".join(lines), id_to_question, answer_dict, a_dict, b_dict

def generate_qa(
                local_random,
                int_max=100,
                int_min=1,
               ):
    x = local_random.randint(int_min, int_max)
    y = local_random.randint(int_min, int_max)
    
    #question = f"which is larger, {x} or {y}?"
    y=x+2
    z=x-6
    w=x-7
    question = f'''The surgeon, who is the boy's father, says 'I cannot operate on this boy, he is my son'. The boy's dad, mom, son, and daughter have {x}, {y}, {z}, and {w} coins. How many coins does the surgeon have?'''

    question = f'''The surgeon, who is the boy's father, says "I cannot operate on this boy, he is my son". How many coins does the surgeon have? Suppose the boy's dad, mom, son, and daughter have {x}, {y}, {z}, and {w} coins.'''

    
    question = f'''The surgeon, who is the boy's father, says "I cannot operate on this boy, he is my son". Suppose this boy's dad and mom have {x} and {y} coins, respectively. How many coins does the surgeon have?'''
    answer = x

    question = f'''The surgeon, who is the boy's father, says "I cannot operate on this boy, he is my son". Who is the doctor to the boy? (A{x}) Father (B{x+1}) Mother (C{x+2}) Son (D{x+3}) Daughter'''
    answer = f'A{x}'

    question = f'''The surgeon, who is the boy's father, says I cannot operate on this boy, he is my son. Who is the doctor to the boy? (A{x}) Father (B{x+1}) Mother\n'''
    answer = f'A{x}'
    return question, answer, question

def convert_row_to_column(id_dict,local_random):
    """
    Generate a text containing a table with two columns: ID and Question.
    The text starts with an introduction, followed by the table rows with ID and Question columns,
    where ID column comes first, followed by the Question column. The last line asks for the answer 
    to the question with a random ID from the dictionary. Also returns the content of the question
    corresponding to the randomly selected ID.
    
    Args:
        id_dict (dict): A dictionary where keys are IDs and values are questions.
    
    Returns:
        tuple: A tuple containing:
            - The generated text with the table and the final question.
            - The dictionary with ID as key and question as value.
            - The content of the randomly selected question.
    """
    intro_text = "Consider a table with two columns, ID and Task. The content of the two columns are as follows. \n\n"
    
    # Generate the ID column
    id_column = "ID\n"
    id_column += "-" * 50 + "\n"
    for _id in id_dict.keys():
        id_column += f"{_id}\n"
    
    # Generate the Question column
    question_column = "\nTask\n"
    question_column += "-" * 50 + "\n"
    for question in id_dict.values():
        question_column += f"{question}\n"
    
    # Randomly select an ID for the last question
    random_id = local_random.choice(list(id_dict.keys()))
    random_question = id_dict[random_id]
    final_question = f"Question: What is the solution to the task with ID {random_id}? "
    
    # Combine everything into the full text
    full_text = intro_text + id_column + question_column + "\n"+final_question
    
    # Return the full text, the dictionary, and the question content for the random ID
    return full_text, id_dict, random_id, random_question