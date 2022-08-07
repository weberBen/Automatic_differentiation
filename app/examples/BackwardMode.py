

import math
import sympy as sym

#%%

import __init__

from ComputationLib.Vector import Vector
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from MathLib.FunctionWrapper import Function
from MathLib.Functions import Sin, sin, Log, log, tan, exp, cos
from MathLib.FunctionReferences import FunctionRef


#%% Example 1

def example1():
    def f(x, y):
        return sin(x)*y + 7*(x*x)

    x = Vector(4, requires_grad=True, label="x")
    y = Vector(7, requires_grad=True, label="y")

    res = f(x,y)
    res.backward()

    cgp = ComputationGraphProcessor(res, human_readable=True)
    cgp.draw(display_nodes_value=True)




#%% Example 2

def example2():
    x = Vector(14.23, requires_grad=True, label="x")
    y = Vector(8, requires_grad=True, label="y")
    z = Vector(6.96, requires_grad=True, label="z")

    X = x*y + 3
    V = sin(y*x) + X + y*log(y, base=2)

    print("result", V)

    V.backward()

    adx_v = x.grad
    ady_v = y.grad
    adz_v = z.grad

    print("grad_x", adx_v)
    print("grad_y", ady_v)
    print("grad_z", adz_v)


    print("----- Manual diff -----")

    mdx_v = math.cos(x.item*y.item)*y.item + y.item
    mdy_v = math.cos(x.item*y.item)*x.item + x.item + (math.log(y.item, math.e)/math.log(2, math.e)) + y.item*(1/(y.item*math.log(2, math.e)))
    mdz_v = 0


    print("grad_x==mdx", adx_v==adx_v)
    print("grad_y==mdy", ady_v==mdy_v)
    print("grad_z==mdz", adz_v==mdz_v)

    cgp = ComputationGraphProcessor(V, human_readable=True)
    cgp.draw(display_nodes_value=True)

    print("----- Rebuilding expression -----")
    print("rebuilt expression from computation graph", cgp.rebuildExpression(track_origin=False))
    print("rebuilt expression from computation graph with node id", cgp.rebuildExpression(track_origin=True))


    print("----- Symbolic diff -----")

    x = sym.Symbol('x')
    y = sym.Symbol('y')
    z = sym.Symbol('z')

    f = sym.sin(y*x) + x*y + 3 + y*sym.log(y,2)

    sdx = sym.diff(f, x)
    sdy = sym.diff(f, y)
    sdz = sym.diff(f, z)

    print("symbolic diff wrt x", sdx)
    print("symbolic diff wrt x", sdy)
    print("symbolic diff wrt x", sdz)

    s_v = {x:14.23, y:8, z:6.96}
    sdx_v = sdx.subs(s_v).evalf()
    sdy_v = sdy.subs(s_v).evalf()
    sdz_v = sdz.subs(s_v).evalf()


    print("grad_x==sdx", adx_v==sdx_v)
    print("grad_y==sdy", ady_v==sdy_v)
    print("grad_z==sdz", adz_v==sdz_v)

#%%

if __name__ == '__main__':
    #example1()
    example2()