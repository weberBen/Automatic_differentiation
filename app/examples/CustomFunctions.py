from importlib_metadata import requires
import numpy as np

#%%
import __init__

from DualNumber import DualNumber
from MathLib.FunctionWrapper import Function
from ComputationLib.Vector import Vector
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from DualNumber import DualNumber
#%%
class NumpyCos(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(np.cos(input_value.primal), - input_value.tangent*np.sin(input_value.primal))

        return np.cos(input_value)
    
    def derivative(self, input_value):
        return -np.sin(input_value)

cos = NumpyCos().apply()

#%%
def f(x,y):
    return  cos(x)*x + 4 + y

_x = np.random.rand(3,2)
_y = np.random.rand(3,2)

def example1():
    x = Vector(_x, requires_grad=True, label="x")
    y = Vector(_y, requires_grad=True, label="y")

    res = f(x,y)
    res.backward()

    print("value", res.item)
    print("grad_x", x.grad)
    print("grad_y", y.grad)

    cgp = ComputationGraphProcessor(res, human_readable=True)
    cgp.draw(display_nodes_value=True)

#%%
def example2():
    x = DualNumber(_x, np.ones((3,2)))
    y = DualNumber(_y, np.zeros((3,2)))

    res = f(x,y)

    print("value", res.primal)
    print("grad_x", res.tangent)

    x = DualNumber(_x, 0)
    y = DualNumber(_y, 1)

    res = f(x,y)

    print("grad_y", res.tangent)

#%%

if __name__ == '__main__':
    example1()
    #example2()