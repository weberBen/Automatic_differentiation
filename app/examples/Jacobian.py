from importlib_metadata import requires
import numpy as np

#%%
import __init__

from DualNumber import DualNumber
from MathLib.Functions import sin, cos, log, tan, exp
from ComputationLib.Vector import Vector
from ComputationLib.utils import computeJacobianBackward
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from DualNumber import DualNumber

#%%

def f(inputs):
    a1 = inputs[0]
    a2 = inputs[1]

    return (a1+sin(a2), (a1+sin(a2)) + a1*a2)

#%%

def example1(display=False):

    inputs = [Vector(2, requires_grad=True), Vector(3, requires_grad=True)]
    res = f(inputs)

    if display:
        cgp = ComputationGraphProcessor(res[0], human_readable=True)
        cgp.draw(display_nodes_value=True, filename="nx1.html")

        cgp = ComputationGraphProcessor(res[0], human_readable=True)
        cgp.draw(display_nodes_value=True, filename="nx2.html")

    jacobian = computeJacobianBackward(inputs, res)

    print(jacobian)


#%%

if __name__ == '__main__':
    example1()