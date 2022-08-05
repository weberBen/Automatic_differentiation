

#%%
from ComputationLib.Vector import Vector

#%%
class Function(object):
    refs = {}

    def __init__(self):
        class_obj =  self.__class__

        self.name = class_obj.__name__

        if class_obj.__name__ not in Function.refs :
            # thread safe since a particular instance is not the target, just the class object without its parameters

            Function.refs[class_obj.__name__] = class_obj # in order to retrieve class by name even if the function has not been defined in that module
    
    def compute(input_value, *argv, **kwargs):
        pass

    def derivative(self, input_value, *argv, **kwargs):
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
