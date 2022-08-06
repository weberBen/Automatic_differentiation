

import math

#%%

import __init__

from ComputationLib.Vector import Vector
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from MathLib.FunctionWrapper import Function
from MathLib.Functions import Sin, sin, Log, log
from MathLib.FunctionReferences import FunctionRef


#%%


x = Vector(14.23, requires_grad=True, label="x")
y = Vector(8, requires_grad=True, label="y")
z = Vector(6.96, requires_grad=True, label="z")

X = x*y + 3
V = sin(y*x) + X + y*log(y, base=2)

print("result", V)

V.backward()

print("grad_x", x.grad)
print("grad_y", y.grad)
print("grad_z", z.grad)

dx = math.cos(x.item*y.item)*y.item + y.item
dy = math.cos(x.item*y.item)*x.item + x.item + math.log(y.item, 2) + y.item*(1/y.item)
dz = 0

print("----- Test -----")
print("grad_x==dx", x.grad==dx)
print("grad_y==dy", y.grad==dy)
print("grad_z==dz", z.grad==dz)

cgp = ComputationGraphProcessor(V, human_readable=True)
cgp.draw(display_nodes_value=True)

print("rebuilt expression from computation graph", cgp.rebuildExpression(track_origin=False))
print("rebuilt expression from computation graph with node id", cgp.rebuildExpression(track_origin=True))

