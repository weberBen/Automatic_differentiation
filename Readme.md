# Automatic Differentiation (AD)

Implementation of a simple automatic differentiation (forward mode with [dual number](https://en.wikipedia.org/wiki/Dual_number) and backward mode with [NetworkX](https://networkx.org/documentation/stable/tutorial.html)) with mathematical explanations.

**The mathematical background of the forward and backward modes is explained in details [here]()**


### Forward mode

Import packages :
```Python
from DualNumber import DualNumber
from MathLib.Functions import sin, cos, tan, log, exp
```

Define a function $f(x,y)$:

```Python
def f(x, y):
    return sin(x)*y + 7*(x*x)
```

Compute the partial derivative $\frac{\partial{f}}{\partial{x}}$

```Python
x = DualNumber(4, 1)
y = DualNumber(7, 0)

res = f(x, y)

f_val = res.primal # f evaluted at (x,y)=(4,7)
df_x = res.tangent # df/dx at (x,y)=(4,7)
```


Compute the partial derivative $\frac{\partial{f}}{\partial{y}}$

```Python
x = DualNumber(4, 0)
y = DualNumber(7, 1)

res = f(x, y)

f_val = res.primal # f evaluted at (x,y)=(4,7)
df_y = res.tangent # df/dy at (x,y)=(4,7)
```

Compute the directionnal derivative $\nabla_{a}f$ for $a=(0.5,3)$

```Python
x = DualNumber(4, 0.5)
y = DualNumber(7, 3)

res = f(x, y)

f_val = res.primal # f evaluted at (x,y)=(4,7)
dir_derivative = res.tangent # directionnal derivative at (x,y)=(4,7) in direction a=(0.5,3)
```

See more examples [here](./app/examples/ForwardMode.py)

## Backward mode

Import packages :
```Python
from ComputationLib.Vector import Vector
from MathLib.Functions import sin, log, tan, exp, cos
```

Define a function :
```Python
def f(x, y):
    return sin(x)*y + 7*(x*x)
```

Define variables $x=7$, $y=7$

```Python
x = Vector(4, requires_grad=True)
y = Vector(7, requires_grad=True)

res = f(x, y)

res.backward()
```

Get the partial derivatives
```Python
dx = x.grad
dy = y.grad
```

You can also display the computation graph :
```Python
from ComputationLib.ComputationGraph import ComputationGraphProcessor

cgp = ComputationGraphProcessor(res, human_readable=True)
cgp.draw(display_nodes_value=True)
```

<p align="center">
    <img
        alt="computation_graph_f"
        src="./assets/images/ghYgsdc8kl.png"
        width="500"
    />
</p>

Self reflective operations ($x \times x$, $\frac{x}{x}$, $x+x$, $x-x$, ...) are represented as a single edge in the graph.

You can set the `label` option when defining new `Vector` to render the variable name during the graph drawing (i.e: `x = Vector(14.23, requires_grad=True, label="x")`).


See more examples [here](./app/examples/BackwardMode.py)

## Custom function

You can define your own functions as following :

```Python
import math
from DualNumber import DualNumber
from MathLib.FunctionWrapper import Function

class MyCos(Function) :
    def __init__(self):
        super().__init__() # needed
    
    def compute(self, input_value):
        if type(input_value) is DualNumber:
            return DualNumber(math.cos(input_value.primal), - input_value.tangent*math.sin(input_value.primal))

        return math.cos(input_value)
    
    def derivative(self, input_value):
        return -math.sin(input_value)

mycos = MyCos().apply()
```

Then :
```Python
mycos(4)
```

The `compute` method is used for the forward mode and for computing the value of the function when evaluating it. The `derivative` is for the backward mode.

## Extending the lib

### Scalar field functions

The lib currently accepts only scalar functions, but by using `numpy` for each standars functions we can evaluate scalar field functions : 

```Python
import numpy as np
from DualNumber import DualNumber
from MathLib.FunctionWrapper import Function

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
```

See more examples [here](./app/examples/CustomFunctions.py)

### Jacobian

The computation of the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of a function $f: (a_1, a_2, ..., a_n) \mapsto (f_1(a_1, ..., a_n), ..., f_m(a_1, ..., a_n))$ from $R^n$ ro $R^m$ is not natively implemented but can be computed on each $f_i$ one by one :

```Python
def computeJacobianBackward(inputs, func_res):
    number_inputs = len(inputs)
    number_outputs = len(func_res)

    jacobian = np.zeros((number_inputs, number_outputs))

    for index, fi in enumerate(func_res):
        fi.backward()

        for a_index, a in enumerate(inputs):
            jacobian[index][a_index] = a.grad
    
    return jacobian
```

See more examples [here](./example/../app/examples/Jacobian.py)

## Limitations

### Thread safe

Thread safe can be implemented by hardening the access to the computation graph for each vector and probably the uuid generation for vector id. But it's not really the purpose of that lib.
