


#%%
import sys
import os
from pathlib import Path

parent_dir = Path(sys.path[0]).parent.absolute()

sys.path.insert(0, '%s%ssrc%sComputationGraph' % (parent_dir, os.sep, os.sep))
sys.path.insert(0, '%s%ssrc%sComputationGraph%sFunctions' % (parent_dir, os.sep, os.sep, os.sep))
sys.path.insert(0, '%s%ssrc' % (parent_dir, os.sep))

#%%

from ComputationGraph import ComputationGraphProcessor
from Vector import Vector
from UniversalNum import UniversalNum
from Function import sin, Function, Log, log, Sin, sin
#%%

import math
#%%

# x = Vector(14.23, required_autograd=True, label="x")
    # y = Vector(2.3, required_autograd=True)
    # z = Vector(7.5, required_autograd=True, label="z")

    # t = Vector(2) + x*y + 6 + UniversalNum.sin(y*x)

    # v = t*2 + UniversalNum.sin(x*y)

x = Vector(14.23, required_autograd=True, label="x")
y = Vector(8, required_autograd=True, label="y")

z =  (x+y) + (x*y) + (x+y)
print("z=", z)
cgp = ComputationGraphProcessor(z)
cgp.draw()

