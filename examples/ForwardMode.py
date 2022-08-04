


#%%
import sys
import os
from pathlib import Path

path = '%s%ssrc' % (Path(sys.path[0]).parent.absolute(), os.sep)
sys.path.insert(0, path)
#%%

from DualNumber import DualNumber
from UniversalNum import UniversalNum

#%%


#%%


def f(x, y):
    return UniversalNum.sin(x)*y + 7*(x*x)

a = (2, 4)

x = DualNumber(a[0], 1) # a_1 + epsilon
y = DualNumber(a[1], 0) # a_2

dx = f(x, y)

x = DualNumber(a[0], 0) # a_1
y = DualNumber(a[1], 1) # a_2 + epsilon

dy = f(x, y)

val = (dx.primal, dy.primal) # dx.primal = dy.primal
grad = (dx.tangent, dy.tangent)

print("value f at ", a, "=", val)
print("grad f at a", a, "=", grad)


# Example : Directionnal derivative
a = (2, 4)
v = (7, 3)

x = DualNumber(a[0], v[0]) # a_1 + v_1*epsilon
y = DualNumber(a[1], v[1]) # a_2 + v_2*epsilon

d_v = f(x, y) # directionnal derivative

print("-"*5)
print("value f at ", a, "=", d_v.primal)
print("directionnal derivative f at ", a, "for direction ", v, "=", d_v.tangent)

