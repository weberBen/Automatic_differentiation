
#%%
import math
#%%
from FunctionWrapper import Function
#%%


class Sin(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        return math.sin(input_value)
    
    def derivative(self, input_value):
        return math.cos(input_value)

sin = Sin().apply()

#%%
class Log(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value, base=10):
        return math.log(input_value, base)
    
    def derivative(self, input_value, base=10):
        return 1/(math.log(base, 2)*input_value)
    

log = Log().apply()
