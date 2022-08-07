import unittest
import math
from importlib_metadata import requires
import sympy as sym
import random
#%%

import __init__

from DualNumber import DualNumber
from MathLib.FunctionWrapper import Function
from MathLib.Functions import Sin, sin, Log, log, cos, tan, exp, Cos, Exp, Log, Tan
from MathLib.FunctionReferences import FunctionRef

from Assertions import IsCloseAssertion

#%%



def evalDifferentiation(expression, vars, direction):

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

        vector = DualNumber(value, direction[var_name])
        symbol = sym.Symbol("_" + var_name)

        locals()[var_name + "_"] = vector
        locals()["_" + var_name] = symbol
    
        f_vector = f_vector.replace("{"+var_name+"}", var_name + "_")
        f_symbol = f_symbol.replace("{"+var_name+"}", "_" + var_name)

    for (func_name, item) in functions.items():
        f_vector = f_vector.replace(func_name, item["v"])
        f_symbol = f_symbol.replace(func_name, item["s"])


    exp_dual = eval(f_vector)
    adv_v = exp_dual.tangent

    sv = eval(f_symbol).subs(vars_symbol).evalf()
    dv = exp_dual.primal

    sdv_v = 0
    for (var_name, value) in vars.items():

        exp_symbol = eval(f_symbol)
        sdv = sym.diff(exp_symbol, locals()["_" + var_name])
        sdv_v += (sdv.subs(vars_symbol).evalf())*direction[var_name]

    
    return {
        "symbolic": {
            "v": sv,
            "d": sdv_v
        },
        "dual": {
            "v": dv,
            "d": adv_v
        }
    }



class ForwardTest(unittest.TestCase, IsCloseAssertion):
    
    def test_sinus(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        x_symbolic = sym.Symbol('x')

        for x in values:
            dual = DualNumber(x, random.uniform(-100, 100))

            self.assertIsClose(math.sin(x), sin(dual).primal, "x: {0}  |  dual: {1}".format(x, dual))
            

            dual = DualNumber(x, 1)
            self.assertIsClose(
                sym.diff(sym.sin(x_symbolic), x_symbolic).subs({"x": x}).evalf(), 
                sin(dual).tangent, 
                "x: {0}  |  dual: {1}".format(x, dual))
            
    def test_cosinus(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        x_symbolic = sym.Symbol('x')

        for x in values:
            dual = DualNumber(x, random.uniform(-100, 100))

            self.assertIsClose(math.cos(x), cos(dual).primal, "x: {0}  |  dual: {1}".format(x, dual))
            
            
            dual = DualNumber(x, 1)
            self.assertIsClose(
                sym.diff(sym.cos(x_symbolic), x_symbolic).subs({"x": x}).evalf(), 
                cos(dual).tangent, 
                "x: {0}  |  dual: {1}".format(x, dual))
    
    def test_tan(self):
        values = [1, -1, 0.89, math.pi, 3, 10.23, math.pi/2, 100, 125] + [random.uniform(-2*math.pi, 2*math.pi) for i in range(100)]

        x_symbolic = sym.Symbol('x')

        for x in values:
            dual = DualNumber(x, random.uniform(-100, 100))

            self.assertIsClose(math.tan(x), tan(dual).primal, "x: {0}  |  dual: {1}".format(x, dual))
            
            
            dual = DualNumber(x, 1)
            self.assertIsClose(
                sym.diff(sym.tan(x_symbolic), x_symbolic).subs({"x": x}).evalf(), 
                tan(dual).tangent, 
                "x: {0}  |  dual: {1}".format(x, dual))
            
    def test_exp(self):
        values = [random.uniform(-100, 100) for i in range(100)]

        x_symbolic = sym.Symbol('x')

        for x in values:
            dual = DualNumber(x, random.uniform(-100, 100))

            self.assertIsClose(math.exp(x), exp(dual).primal, "x: {0}  |  dual: {1}".format(x, dual))
            
            
            dual = DualNumber(x, 1)
            self.assertIsClose(
                sym.diff(sym.exp(x_symbolic), x_symbolic).subs({"x": x}).evalf(), 
                exp(dual).tangent, 
                "x: {0}  |  dual: {1}".format(x, dual))
            
    def test_log(self):
        values = [random.uniform(0.1, 100) for i in range(100)]

        x_symbolic = sym.Symbol('x')

        for base in [2, math.e, 10, 14]:
            for x in values:
                dual = DualNumber(x, random.uniform(-100, 100))

                self.assertIsClose(math.log(x, base), log(dual, base).primal, "x: {0}  |  dual: {1}  |  base: {2}".format(x, dual, base))
                
                
                dual = DualNumber(x, 1)
                self.assertIsClose(
                    sym.diff(sym.log(x_symbolic, base), x_symbolic).subs({"x": x}).evalf(), 
                    log(dual, base).tangent, 
                    "x: {0}  |  dual: {1}  |  base: {2}".format(x, dual, base))
    

    def test_grad1(self):
        exp = "sin({y}*{x}) + {x}+{y}+3 + {y}*log({y}, 2) + {y}/{x} - {x}*{x}"
        vars = {
            "x": 14.23,
            "y": 8,
            "z": 6.96
        }

        direction = {
            "x": 1,
            "y": 0,
            "z": 0
        }

        res = evalDifferentiation(exp, vars, direction)

        msg = "expression: {0}  |  vars: {1}".format(exp, vars)
        self.assertIsClose(res["symbolic"]["v"], res["dual"]["v"], msg)


        msg = "expression: {0}  |  vars: {1}  |  direction: {2}".format(exp, vars, direction)
        self.assertIsClose(res["symbolic"]["d"], res["dual"]["d"], msg)


        direction = {
            "x": 0,
            "y": 1,
            "z": 0
        }

        res = evalDifferentiation(exp, vars, direction)

        msg = "expression: {0}  |  vars: {1}  |  direction: {2}".format(exp, vars, direction)
        self.assertIsClose(res["symbolic"]["d"], res["dual"]["d"], msg)
       

        direction = {
            "x": 0,
            "y": 0,
            "z": 1
        }

        res = evalDifferentiation(exp, vars, direction)

        msg = "expression: {0}  |  vars: {1}  |  direction: {2}".format(exp, vars, direction)
        self.assertIsClose(res["symbolic"]["d"], res["dual"]["d"], msg)
    

        direction = {
            "x": 7,
            "y": 4.36,
            "z": 12
        }

        res = evalDifferentiation(exp, vars, direction)

        msg = "expression: {0}  |  vars: {1}  |  direction: {2}".format(exp, vars, direction)
        self.assertIsClose(res["symbolic"]["d"], res["dual"]["d"], msg)

if __name__ == '__main__':

    unittest.main()
