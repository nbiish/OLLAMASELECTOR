DEFAULT_MODEL = "gpt-4o-2024-05-13"
# Root Class for Module
class Module(object):
    def __init__(self,
                 max_tokens = 1000,
                 model = DEFAULT_MODEL,
                ):
        self.model = model
        self.max_tokens = max_tokens
        
    def setmodel(self,model=DEFAULT_MODEL):
        self.model = model
        return

    def get_response(self,query):
        return Get_Generate(query,self.model)