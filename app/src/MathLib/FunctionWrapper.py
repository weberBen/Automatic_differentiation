

#%%
from ComputationLib.Vector import Vector
from .FunctionReferences import FunctionRef

#%%
class Function(object):
    def __init__(self):
        class_obj =  self.__class__

        self.name = class_obj.__name__

        # regsiter function by name to retrieve them later in the gradient computation to 
        # instantiate a new object and have access to the derivative function
        # while the computation with the apply function return a float/int with no changes in the vector code
        FunctionRef.add(class_obj)
    
    def compute(input_value, *argv, **kwargs):
        # Need to be implemented by children classes
        # input_value is int/float
        pass

    def derivative(self, input_value, *argv, **kwargs):
        # Need to be implemented by children classes
        # input_value is int/float
        pass
    
    def _compute(self, input_value, *argv, **kwargs):
        if type(input_value) is Vector:
            return input_value.__apply__(self.name, self.compute, *argv, **kwargs)
        
        return self.compute(input_value, *argv, **kwargs)

    def _derivative(self, input_value, *argv, **kwargs):
        if type(input_value) is Vector:
            return input_value.__apply__(self.name, self.derivative, *argv, **kwargs)

        return self.derivative(input_value, *argv, **kwargs)

    def _apply(self, input_value, *argv, **kwargs):
        return self._compute(input_value, *argv, **kwargs)
    
    def apply(self):
        return self._apply
    
    def __str__(self):
         return "Function({0})".format(self.name)
