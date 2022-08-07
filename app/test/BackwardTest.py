import unittest
import math
from importlib_metadata import requires
import sympy as sym
import random
#%%

import __init__

from ComputationLib.Vector import Vector
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from MathLib.FunctionWrapper import Function
from MathLib.Functions import Sin, sin, Log, log, cos, tan, exp, Cos, Exp, Log, Tan
from MathLib.FunctionReferences import FunctionRef

from Assertions import IsCloseAssertion
#%%


def evalDifferentiation(expression, vars):

    functions = {
        "log": {
            "v": "log",
            "s": "sym.log"
        },
        "cos": {
            "v": "cos",
            "s": "sym.cos"
        },
        "sin": {
            "v": "sin",
            "s": "sym.sin"
        },
        "exp": {
            "v": "exp",
            "s": "sym.exp"
        },
        "tan": {
            "v": "tan",
            "s": "sym.tan"
        }
    }

    vars_vector = {}
    vars_symbol = {}

    f_vector = expression
    f_symbol = expression

    for (var_name, value) in vars.items():
        
        if (var_name.find("{") != -1) or (var_name.find("}") != -1):
            msg = "Invalid var name '{0}'".format(var_name)
            raise Exception(msg)
        
        vars_vector[var_name + "_"] = value
        vars_symbol["_" + var_name] = value

        vector = Vector(value, requires_grad=True, label=var_name)
        symbol = sym.Symbol("_" + var_name)

        locals()[var_name + "_"] = vector
        locals()["_" + var_name] = symbol
    
        f_vector = f_vector.replace("{"+var_name+"}", var_name + "_")
        f_symbol = f_symbol.replace("{"+var_name+"}", "_" + var_name)

    for (func_name, item) in functions.items():
        f_vector = f_vector.replace(func_name, item["v"])
        f_symbol = f_symbol.replace(func_name, item["s"])


    exp_vector = eval(f_vector)
    exp_vector.backward()

    output = []
    for (var_name, value) in vars.items():

        exp_symbol = eval(f_symbol)
        sdv = sym.diff(exp_symbol, locals()["_" + var_name])
        sdv_v = sdv.subs(vars_symbol).evalf()

        
        vector = locals()[var_name + "_"]
        adv_v = vector.grad

        output.append({
            "wrt": var_name,
            "symbolic": sdv_v,
            "vector": adv_v
        })
    
    return output


class BackwardTest(unittest.TestCase, IsCloseAssertion):
    
    def test_sinus(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        for x in values:
            self.assertEqual(math.sin(x), sin(x), "x={0}".format(x))

    def test_dsinus(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        v = sym.Symbol('v')
        f = sym.sin(v)

        for x in values:
            dv_ref = sym.diff(f, v).subs({v: x}).evalf()
            dv = Sin()._derivative(x)

            # check if isClose instead of equals because of symPy floating precision that does not match math lib precision
            self.assertIsClose(dv_ref, dv, msg="x={0}".format(x))
    
    def test_cosinus(self):
        
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        for x in values:
            self.assertEqual(math.cos(x), cos(x), "x={0}".format(x))
    
    def test_dcosinus(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        v = sym.Symbol('v')
        f = sym.cos(v)

        for x in values:
            dv_ref = sym.diff(f, v).subs({v: x}).evalf()
            dv = Cos()._derivative(x)
            
            # check if isClose instead of equals because of symPy floating precision that does not match the math lib precision
            self.assertIsClose(dv_ref, dv, msg="x={0}".format(x))

    def test_tan(self):
        
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125, -2*math.pi] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        for x in values:
            self.assertEqual(math.tan(x), tan(x), "x={0}".format(x))
    
    def test_dtan(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        v = sym.Symbol('v')
        f = sym.tan(v)

        for x in values:
            dv_ref = sym.diff(f, v).subs({v: x}).evalf()
            dv = Tan()._derivative(x)
            
            self.assertIsClose(dv_ref, dv, "x={0}".format(x))

    def test_exp(self):
        
        values = [random.uniform(-100, 100) for i in range(100)]

        for x in values:
            self.assertEqual(math.exp(x), exp(x), "x={0}".format(x))

    def test_dexp(self):
        values = [random.uniform(-100, 100) for i in range(100)]

        v = sym.Symbol('v')
        f = sym.exp(v)

        for x in values:
            dv_ref = sym.diff(f, v).subs({v: x}).evalf()
            dv = Exp()._derivative(x)
            
            self.assertIsClose(dv_ref, dv, "x={0}".format(x))
    
    def test_log(self):
        
        values = [random.uniform(0.1, 100) for i in range(100)]

        for base in [2, math.e, 10, 14]:
            for x in values:
                self.assertEqual(math.log(x, base), log(x, base), "x: {0}  |  base: {1}".format(x, base))
    
    def test_dlog(self):
        values = [random.uniform(0.1, 100) for i in range(100)]

        for base in [2, math.e, 10, 14]:
            v = sym.Symbol('v')
            f = sym.log(v, base)

            for x in values:
                dv_ref = sym.diff(f, v).subs({v: x}).evalf()
                dv = Log()._derivative(x, base)
                
                self.assertIsClose(dv_ref, dv, "x: {0}  |  base: {1}".format(x, base))
    
    def test_grad1(self):

        exp = "sin({y}*{x}) + {x}+{y}+3 + {y}*log({y}, 2)"
        vars = {
            "x": 14.23,
            "y": 8,
            "z": 6.96
        }

        for item in evalDifferentiation(exp, vars):
            msg = "expression: {0}  | vars: {1}  | wrt: {2}".format(exp, vars, item["wrt"])
            self.assertIsClose(item["symbolic"], item["vector"], msg)
        
    def test_grad2(self):

        exp = "{x}/({y}*log(14*{x})) - 4+{x}/{y} + {z}"
        vars = {
            "x": 14.23,
            "y": 8,
            "z": 6.96
        }

        for item in evalDifferentiation(exp, vars):
            msg = "expression: {0}  | vars: {1}  | wrt: {2}".format(exp, vars, item["wrt"])
            self.assertIsClose(item["symbolic"], item["vector"], msg)
    
    def test_grad3(self):

        exp = "cos((exp({x})+{y})/({z}*{z}*{z}))"
        vars = {
            "x": 14.23,
            "y": 8,
            "z": 6.96
        }

        for item in evalDifferentiation(exp, vars):
            msg = "expression: {0}  | vars: {1}  | wrt: {2}".format(exp, vars, item["wrt"])
            self.assertIsClose(item["symbolic"], item["vector"], msg)
    
    def test_vector1(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)

        # check if x*y and y*x produce the same node in the computation graph
        a = x*y + y*x

        (G, mapping) = a._getCleanComputationGraph()

        
        node_x = G.nodes[x.id]
        node_y = G.nodes[y.id]

        nodes = list(G.successors(x.id))
        self.assertTrue(len(nodes)==1)

        nodes = list(G.successors(y.id))
        self.assertTrue(len(nodes)==1)

        child_id = nodes[0]
        nodes = list(G.predecessors(child_id))
        self.assertTrue(len(nodes)==2)
        
        self.assertTrue((x.id in nodes) and (y.id in nodes))


    def test_vector2(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)

        a = x*y + 3
        b = sin(x*y)

        (G, mapping) = b._getCleanComputationGraph()

        
        node_x = G.nodes[x.id]
        node_y = G.nodes[y.id]

        nodes = list(G.successors(x.id))
        self.assertTrue(len(nodes)==1)

        nodes = list(G.successors(y.id))
        self.assertTrue(len(nodes)==1)

        child_id = nodes[0]
        nodes = list(G.predecessors(child_id))
        self.assertTrue(len(nodes)==2)
        
        self.assertTrue((x.id in nodes) and (y.id in nodes))
    
    def test_vector3(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)

        b = x*y + 3 + sin(x*y)

        (G, mapping) = b._getCleanComputationGraph()

        
        node_x = G.nodes[x.id]
        node_y = G.nodes[y.id]

        nodes = list(G.successors(x.id))
        self.assertTrue(len(nodes)==1)

        nodes = list(G.successors(y.id))
        self.assertTrue(len(nodes)==1)

        child_id = nodes[0]
        nodes = list(G.predecessors(child_id))
        self.assertTrue(len(nodes)==2)
        
        self.assertTrue((x.id in nodes) and (y.id in nodes))
    
    def test_vector4(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        
        # check if unused nodes are removed
        a = z*3 + sin(z) + y
        b = sin(x*y)

        (G, mapping) = b._getCleanComputationGraph()

        
        self.assertTrue(z.id not in list(G.nodes()))

        self.assertTrue(len(list(G.nodes()))==4)
    
    def test_vector5(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        

        b = 4 - sin(x*y)

        (G, mapping) = b._getCleanComputationGraph()

        
        self.assertTrue(z.id not in list(G.nodes()))
        
        self.assertTrue(len(list(G.nodes()))==6)
    
    def test_vector7(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        

        b = -sin(x*y) + 4

        (G, mapping) = b._getCleanComputationGraph()

        
        self.assertTrue(z.id not in list(G.nodes()))

        self.assertTrue(len(list(G.nodes()))==8)
    
    def test_vector8(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        

        b = x*y + 4

        (G, mapping) = b._getCleanComputationGraph()
        
        self.assertTrue(z.id not in list(G.nodes()))
        
        self.assertTrue(len(list(G.nodes()))==5)
    
    def test_vector9(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        
        # check if unused nodes are removed
        b = 4 + x*y

        (G, mapping) = b._getCleanComputationGraph()

        
        self.assertTrue(z.id not in list(G.nodes()))
        
        self.assertTrue(len(list(G.nodes()))==5)
    
    def test_vector9(self):
        x = Vector(14.23, requires_grad=True)
        y = Vector(8, requires_grad=True)
        z = Vector(6.96, requires_grad=True)
        
        a = x+y*1 + z
        b = 4*y

        (G, mapping) = b._getCleanComputationGraph()

        
        self.assertTrue(z.id not in list(G.nodes()))
        self.assertTrue(x.id not in list(G.nodes()))
        
        self.assertTrue(len(list(G.nodes()))==3)


       

if __name__ == '__main__':

    unittest.main()
