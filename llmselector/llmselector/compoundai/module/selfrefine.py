from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter

DESCRIPTION_SELFREFINE = '''This compound AI system uses 3 modules to solve a question. The module 0 generates an initial answer, module 1 gives some feedback to this initial answer. Finally, module 2 uses the initial answer and feedback to generate an updated answer.
'''
PROMPT_TEMPLATE_FEEDBACK = "Below is a question and an initial answer. Is the predicted code output correct? If not, explain which reasoning steps in the initial answer leads to the mistakes. \nOriginal question:{task}\nModel answer:{answer}\nFeedback:"
PROMPT_TEMPLATE_REFINE =  "The below is a question, an initial answer, and some feedback. Generate a new step-by-step answer based on the feedback. Make sure that you fix all mistakes identified by the feedback.\nOriginal question:{task}\nInitial answer:{answer}\nFeedback:{feedback}\nNew answer:"

class Refiner(Module):
    def __init__(self,
                prompt_template_refine= PROMPT_TEMPLATE_REFINE,
                ):
        self.prompt_template_refine = prompt_template_refine
        self.model = DEFAULT_MODEL
        return
        
    def get_response(self,task, answer, feedback):
        refine_prompt = self.prompt_template_refine.format(
            task=task,feedback=feedback,answer=answer)
        return Get_Generate(refine_prompt,self.model) 
    
class Critic(Module):
    def __init__(self,
                prompt_template_feedback= PROMPT_TEMPLATE_FEEDBACK,
                ):
        self.prompt_template_feedback = prompt_template_feedback
        self.model = DEFAULT_MODEL
        return
    def get_response(self, task, answer):
        feedback_prompt = self.prompt_template_feedback.format(
            task=task,answer=answer)
        return Get_Generate(feedback_prompt,self.model) 

class Generator(Module):
    def get_response(self,query):
        return Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )

class SelfRefine(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_SELFREFINE,
                ):
        super().__init__(description=description)
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [Generator(),[0]],
                           [Critic(prompt_template_feedback=PROMPT_TEMPLATE_FEEDBACK),[0,1]],
                           [Refiner(prompt_template_refine=PROMPT_TEMPLATE_REFINE), [0,1,2]],
                           ]
        return pipeline
