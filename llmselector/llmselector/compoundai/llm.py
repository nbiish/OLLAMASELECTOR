from itertools import product
from tqdm import tqdm
import random
#from .compoundai import Get_Generate
import re
from tqdm import tqdm
import concurrent.futures

import os
from sqlitedict import SqliteDict
import google.generativeai as genai
import time

# Make tqdm work with pandas apply
tqdm.pandas()

os.environ['TASK_NAME'] = 'test'
os.environ['DB_PATH'] = 'test.sqlite'

os.environ['OPENAI_API_KEY'] = "openai_api_key"
os.environ['ANTHROPIC_API_KEY'] = "anthropic_api_key" 
os.environ['TOGETHER_API_KEY'] = "together_ai_api_key"
os.environ['GEMINI_API_KEY'] = "gemini_api_key"

def Get_Generate(prompt, model_gen,stop=None,
                max_tokens=1000,
                 temperature=0.1,
                ):
    max_attempts = 20
    attempt = 0
    #print(f"--models and stop {model_gen} and {stop}-- ")
    if model_gen in model_provider_mapper:
        Provider = model_provider_mapper[model_gen]
    else:
        Provider = MyAPI_OpenAI

    '''
    return Provider.get_response(text=prompt, model=model_gen,
                                         stop=stop)
    '''
    while attempt < max_attempts:
        try:
            return Provider.get_response(text=prompt, 
                                         model=model_gen,
                                         stop=stop,
                                        max_tokens=max_tokens,
                                         temperature=temperature,
                                        )
        except Exception as e:
            attempt += 1
            if attempt < max_attempts:
                print(f"Attempt {attempt} failed with error: {e}. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Attempt {attempt} failed with error: {e}. No more retries left.")
                raise


## Lower level LLM services
from openai import OpenAI

import zlib, pickle, sqlite3
def my_encode(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def my_decode(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

#tentative_db = SqliteDict("temp.sqlite",encode=my_encode, decode=my_decode)

def extract_db(key, value):
    return # remove this if you want to extract the cache
    tentative_db[key] = value
    tentative_db.commit()
    return


class VLMService(object):
    def __init__(self, db_name='all'):
        dbname = os.getenv('DB_PATH').format(name=db_name)
        #print("dbname",dbname)
        self.db = SqliteDict(dbname,encode=my_encode, decode=my_decode)
#        self.db = SqliteDict(os.getenv('DB_PATH').format(name=db_name))

        return
    
    def setup_db():
        dbname = os.getenv('DB_PATH').format(name=db_name)
        #print("dbname",dbname)
        self.db = SqliteDict(dbname,encode=my_encode, decode=my_decode)
        pass

    def get_response(self, text, model='gpt-4o'):
        return

    def get_cache_key(self,
                     text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                      stop=None,
                     ):
        if(stop is None):
            key = f"{model}_{text}_{max_tokens}_{temperature}"
        else:
            key = f"{model}_{text}_{max_tokens}_{temperature}_{stop}"
        #print(f"key is __{key}__")
        return key
        
    def load_response(self,
                      text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                      stop=None,
                     ):
        key = self.get_cache_key(text=text,model=model,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 stop=stop)
        #print("key length", len(key))

        #if(key not in self.db):
            #print("key is not found!")
            #print(f"key is _{key}_")
        result = self.db[key]
        extract_db(key,result)
        return result

    def save_response(self,
                      text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                      stop=None,
                      response ="",
                     ):
        key = self.get_cache_key(text=text,model=model,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 stop=stop)
        self.db[key] = response
        self.db.commit()
        result = self.db[key]
        return True
        
class OpenAILLMService(VLMService):
    def __init__(self,db_name='all'):
        super(OpenAILLMService, self).__init__(db_name)        
        client = OpenAI()
        self.client =client
        return
        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                    ):
        #print(f"--stop is {stop}--")
        try: 
            return self.load_response(text=text,model=model,max_tokens=max_tokens,temperature=temperature,
                                     stop=stop)
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass
        if(not(stop is None)):
            response = self.client.chat.completions.create(
          model=model,
          n=1,
          stop=stop,
        messages=[
            {
          "role": "user",
          "content": [
            {"type":"text", 
             "text":f"{text}"
            },
          ],
        }
      ],          
      max_tokens=max_tokens,
      #max_completion_tokens=max_tokens,

        temperature=temperature,
    )
        else:
            response = self.client.chat.completions.create(
          model=model,
          n=1,          
            messages=[
            {
          "role": "user",
          "content": [
            {"type": "text", 
             "text": f"{text}"},
          ],
        }
      ],
      max_tokens=max_tokens,
      #max_completion_tokens=max_tokens,

    temperature=temperature,
    )
        self.save_response(text=text,
                           model=model,max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=response.choices[0].message.content)

        return response.choices[0].message.content


import anthropic

class AnthropicLLMService(VLMService):
    def __init__(self,db_name='all'):
        super(AnthropicLLMService, self).__init__(db_name)        
        self.client = anthropic.Anthropic()
        return
        
    def get_response(self, text, 
                     model='claude-3-opus-20240229',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                    ):
        '''
        #print("loading keys...")
        res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
        #print("load key success")
        '''
        #print(f"stop is --{stop}--")
        #print("text is",text)
        try: 
            res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
            #print("load cached!")
            return res
        except:
            #print(f"load cache failed on {model}. Generate new")
            #print(f"load failed on {model} with text _{text}__, directly run-")
            try:
                response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop,
            messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{text}"
                    }
                ],
            }
        ],
            )
            except:
                response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{text}"
                    }
                ],
            }
        ],
            )
                
            self.save_response(text=text,
                               model=model,
                               max_tokens=max_tokens,
                               stop=stop,
                               temperature=temperature,response=response.content[0].text)
            return response.content[0].text



from together import Together


class TogetherAILLMService(VLMService):
    def __init__(self,db_name='all'):
        super(TogetherAILLMService, self).__init__(db_name)        
        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
        return
        
    def get_response(self, text, 
                     model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                    ):
        '''
        res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
        '''
        try: 
            res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
            #print("load cached!")
            return res
        except:
            #print(f"load failed on {model} with text _{text}__, directly run-")

            client = self.client

            # Prepare the message context
            messages = [{"role": "user", "content": text}]
        
            # Make the API call to the LLaMA model
            response = client.chat.completions.create(
            model=model,
            #model = 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=stop,
            stream=False
            )
            res = response.choices[0].message.content
                
            self.save_response(text=text,
                               model=model,
                               max_tokens=max_tokens,
                               stop=stop,
                               temperature=temperature,response=res)
            return res


class GeminiLLMService(VLMService):
    def __init__(self,db_name='all'):
        super(GeminiLLMService, self).__init__(db_name)        
        client = OpenAI()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client =client
        return
        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                    ):
        #print(f"--stop is {stop}--")
        try: 
            return self.load_response(text=text,model=model,max_tokens=max_tokens,temperature=temperature,
                                     stop=stop)
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass

        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
            }
        model_instance = genai.GenerativeModel(
            model_name=model,
        generation_config=generation_config,
        )
        chat_session = model_instance.start_chat(
          history=[
            ]
        )
        response = chat_session.send_message(text)
        output_text = response.candidates[0].content.parts[0].text
        self.save_response(text=text,
                           model=model,max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=output_text)
        return output_text

'''
MyAPI_OpenAI = OpenAILLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Anthropic = AnthropicLLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Together = TogetherAILLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Google = GeminiLLMService(db_name=os.getenv('TASK_NAME'))

model_provider_mapper={
    "claude-3-opus-20240229":MyAPI_Anthropic,
    "gpt-4-turbo-2024-04-09":MyAPI_OpenAI,
    "gpt-4o-mini-2024-07-18":MyAPI_OpenAI,
    'gpt-4o-2024-05-13':MyAPI_OpenAI,
    'gpt-4o-2024-08-06':MyAPI_OpenAI,

    "o1-mini-2024-09-12":MyAPI_OpenAI,
    'claude-3-5-sonnet-20240620':MyAPI_Anthropic,
    'claude-3-haiku-20240307':MyAPI_Anthropic,
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo':MyAPI_Together,
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo':MyAPI_Together,
    'databricks/dbrx-instruct':MyAPI_Together,
    'Qwen/Qwen2.5-72B-Instruct-Turbo':MyAPI_Together,
    'mistralai/Mixtral-8x22B-Instruct-v0.1':MyAPI_Together,
    'gemini-1.5-pro':MyAPI_Google,
    'gemini-1.5-flash':MyAPI_Google, 
    'gemini-2.0-flash-exp':MyAPI_Google,
    'gemini-1.5-flash-8b':MyAPI_Google,
}
'''

MyAPI_OpenAI = None
MyAPI_Anthropic = None
MyAPI_Together = None
MyAPI_Google = None
model_provider_mapper = {}

# Define the model_provider_mapper
def initialize_services(task_name="test"):
    global MyAPI_OpenAI, MyAPI_Anthropic, MyAPI_Together, MyAPI_Google, model_provider_mapper

    # Initialize LLM services based on task_name
    MyAPI_OpenAI = OpenAILLMService(db_name=task_name)
    MyAPI_Anthropic = AnthropicLLMService(db_name=task_name)
    MyAPI_Together = TogetherAILLMService(db_name=task_name)
    MyAPI_Google = GeminiLLMService(db_name=task_name)

    # Update model_provider_mapper
    model_provider_mapper = {
        "claude-3-opus-20240229": MyAPI_Anthropic,
        "gpt-4-turbo-2024-04-09": MyAPI_OpenAI,
        "gpt-4o-mini-2024-07-18": MyAPI_OpenAI,
        'gpt-4o-2024-05-13': MyAPI_OpenAI,
        'gpt-4o-2024-08-06': MyAPI_OpenAI,
        "o1-mini-2024-09-12": MyAPI_OpenAI,
        'claude-3-5-sonnet-20240620': MyAPI_Anthropic,
        'claude-3-haiku-20240307': MyAPI_Anthropic,
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': MyAPI_Together,
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': MyAPI_Together,
        'databricks/dbrx-instruct': MyAPI_Together,
        'Qwen/Qwen2.5-72B-Instruct-Turbo': MyAPI_Together,
        'mistralai/Mixtral-8x22B-Instruct-v0.1': MyAPI_Together,
        'gemini-1.5-pro': MyAPI_Google,
        'gemini-1.5-flash': MyAPI_Google, 
        'gemini-2.0-flash-exp': MyAPI_Google,
        'gemini-1.5-flash-8b': MyAPI_Google,
    }

# Initialize services when the module is loaded
initialize_services()


 