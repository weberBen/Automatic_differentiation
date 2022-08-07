
#%%
import math
#%%
from .FunctionWrapper import Function

from DualNumber import DualNumber
#%%


class Sin(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(math.sin(input_value.primal), input_value.tangent*math.cos(input_value.primal))
        
        return math.sin(input_value)
    
    def derivative(self, input_value):
        return math.cos(input_value)

sin = Sin().apply()

#%%

class Cos(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(math.cos(input_value.primal), - input_value.tangent*math.sin(input_value.primal))

        return math.cos(input_value)
    
    def derivative(self, input_value):
        return -math.sin(input_value)

cos = Cos().apply()

#%%

class Tan(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(math.tan(input_value.primal), - input_value.tangent/(math.cos(input_value.primal)**2))

        return math.tan(input_value)
    
    def derivative(self, input_value):
        return (math.cos(input_value)**2+math.sin(input_value)**2)/(math.cos(input_value)**2)

tan = Tan().apply()

#%%

class Exp(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(math.exp(input_value.primal), input_value.tangent*math.exp(input_value.primal))

        return math.exp(input_value)
    
    def derivative(self, input_value):
        return math.exp(input_value)

exp = Exp().apply()

#%%
class Log(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value, base=10):
        if type(input_value) is DualNumber:
            return DualNumber(math.log(input_value.primal, base), input_value.tangent/input_value.primal)
        
        return math.log(input_value, base)
    
    def derivative(self, input_value, base=10):
        return 1/(math.log(base, math.e)*input_value)
    

log = Log().apply()
