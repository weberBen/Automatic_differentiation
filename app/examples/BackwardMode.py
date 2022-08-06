

import math

#%%

import __init__

from ComputationLib.Vector import Vector
from ComputationLib.ComputationGraph import ComputationGraphProcessor
from MathLib.FunctionWrapper import Function
from MathLib.Functions import Sin, sin, Log, log
from MathLib.FunctionReferences import FunctionRef


#%%

# x = Vector(14.23, required_autograd=True, label="x")
    # y = Vector(2.3, required_autograd=True)
    # z = Vector(7.5, required_autograd=True, label="z")

    # t = Vector(2) + x*y + 6 + UniversalNum.sin(y*x)

    # v = t*2 + UniversalNum.sin(x*y)


x = Vector(14.23, requires_grad=True, label="x")
y = Vector(8, requires_grad=True, label="y")
z = Vector(6.96, requires_grad=True, label="z")
#z = log(x, base=14) + x*y + y*sin(y) + x*y

d = y+x+3
v = x+y + z

print("z=", v)

# gradient = v.backward()

# print("gradx=", x.grad)
# print("grady=", y.grad)
# print("grady=", z.grad)
# print("gradient=", gradient)

cgp = ComputationGraphProcessor(v, human_readable=False)
cgp.draw(display_nodes_value=True)

# (G, mapping) = z._getCleanComputationGraph(human_readable=True)

# for (node_id, data) in G.nodes(data=True):
#     if ("type" in data) and data["type"]=="function":
#         print("d=", data)

#         v = FunctionRef.get(data["func_name"])()._compute(data["input_value"], *data["func_argv"], **dict(data["func_kwargs"]))
#         dv = FunctionRef.get(data["func_name"])()._derivative(data["input_value"], *data["func_argv"], **dict(data["func_kwargs"]))
#         print("v=", v, "dv=", dv)

# import networkx as nx
# G = nx.MultiDiGraph()

# G.add_node("A", label="oki")
# G.add_node("B")
# G.add_node("C")

# G.add_edge("A", "B", type="operation", operator="+")
# G.add_edge("A", "B", type="operation", operator="-")
# G.add_edge("A", "B", type="operation", operator="+")

# G.add_edge("A", "C", type="operation", operator="+")

# G.add_edge("A", "C", type="function", func_name="sin")

# H = nx.MultiDiGraph()
# H.add_node("A")
# H.add_node("B")
# H.add_node("D")
# H.add_node("E")
# H.add_node("V")

# H.add_edge("A", "B", type="operation", operator="/")
# H.add_edge("D", "E", type="operation", operator="/")

# G.add_edges_from(list(H.edges(data=True)))
# G.add_nodes_from(list(H.nodes(data=True)))
# #     left.computation_graph.add_nodes_from(right.computation_graph.nodes(data=True))


# for (node_from, node_to, data) in G.edges(data=True):
#     print(node_from, node_to, data)

# for (key, item) in  G.get_edge_data("A", "B", default={}).items():
#     print(key, item)

# G.add_node("G", **{"type": "ok", "operator": "p"})

# print(G.nodes(data=True))
