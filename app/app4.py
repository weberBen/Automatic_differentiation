from __future__ import annotations
from email.policy import default
from hashlib import new
from stat import SF_APPEND
from typing import NewType
import math
import weakref
import uuid
import json
import networkx as nx
import matplotlib.pyplot as plt
import sys

# x = linespace(100, 200)
# y = a*x + b*x² + sin(x)

#%%
from src.DualNumber import DualNumber
from src.UniversalNum import UniversalNum


#%%





if __name__ == "__main__":

    # from pyvis.network import Network

    # def func(x, y, z):
    #     return x*y + z + Vector(0.2)
    
    # x = Vector(14.23, required_autograd=True, label="x")
    # y = Vector(2.3, required_autograd=True)
    # z = Vector(7.5, required_autograd=True, label="z")

    # t = Vector(2) + x*y + 6 + UniversalNum.sin(y*x)

    # v = t*2 + UniversalNum.sin(x*y)

    #v = 3*t + UniversalNum.sin(y*x) + y*x + 5*UniversalNum.sin(x*y)
    #v = t + Vector(16) + x*x  + (UniversalNum.sin(y*x) + Vector(4))*x + UniversalNum.sin((y*x)) + (Vector(16)+Vector(3))
    #v = x*y + y*x #-> un seul edge commme le x*x A AJOUTER au moment de rebuildExpression
    #v = UniversalNum.sin(x*y) + 2 + UniversalNum.sin(y*x)

    # t = Vector(2) + x*y + 6
    # v = x
    # quand on demande l'expression de v c'est en fait l'expression de x car on ne peut pas override l'operateur d'assignement = avec Python

    ###########################################################################

    
    # computation_graph_processor = ComputationGraphProcessor(v, human_readable=True)
    # print("val string=", computation_graph_processor.rebuildExpression(track_origin=True))
    # computation_graph_processor.draw()

    # G = v._getCleanComputationGraph(human_readable=True)

    # pos = nx.kamada_kawai_layout(G)

    # edge_labels = dict([((node_from, node_to), (f'{data["type"]}' if "type" in data else ""))
    #                 for node_from, node_to, data in G.edges(data=True)])
    
    # def formatNodeLabel(node, data):
    #     if "type" not in data:
    #         if "value" in data:
    #             return str(data["value"])
    #         return ""
        
    #     if data["type"]=="operator":
    #         return data["operator"]
        
    #     if data["type"]=="constant":
    #         return f'const({data["value"]})'
        
    #     if data["type"]=="variable":
    #         return f'var({data["value"]})={data["name"]}'
        
    #     if data["type"]=="function":
    #         return data["func_name"]
        
    #     return str(data["type"])
    

    # node_labels = dict([(node, str(node) + ":" + formatNodeLabel(node, data))
    #                 for node, data in G.nodes(data=True)])

    # #%%

    # nx.draw_networkx(G, pos, with_labels = True, labels=node_labels)

    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # #plt.show()


    # nt = Network('100%', '100%', directed=True, layout=False)
    # for (node_id, data) in G.nodes(data=True):
    #     nt.add_node(node_id, label=formatNodeLabel(node_id, data), group=data["type"], title=f'id({node_id})')
    
    # for (node_from_id, node_to_id, data) in G.edges(data=True):
    #     nt.add_edge(node_from_id, node_to_id, physics=True)

    # nt.show('nx.html')


    # nx_graph = nx.cycle_graph(10)
    # nx_graph.nodes[1]['title'] = 'Number 1'
    # nx_graph.nodes[1]['group'] = 1
    # nx_graph.nodes[3]['title'] = 'I belong to a different group!'
    # nx_graph.nodes[3]['group'] = 10
    # nx_graph.add_node(20, size=20, title='couple', group=2)
    # nx_graph.add_node(21, size=15, title='couple', group=2)
    # nx_graph.add_edge(20, 21, weight=5)
    # nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    # nt = Network('500px', '500px')
    # # populates the nodes and edges data structures
    # nt.from_nx(nx_graph)
    # nt.show('nx.html')



    #v1 = DualNumber(1, 2)
    #v2 = DualNumber(2, 3)
    #v3 = UniversalNum.sin(v2)


    #print(v_3)

    # g = nx.MultiDiGraph()

    # g.add_node("A")

    # if not g.has_node("B"):
    #     g.add_node("B")
    
    # g.add_edge("A", "B", type="+")
    # g.add_edge("A", "A", type="operation", operator="+", value=145)
    # g.add_edge("A", "A", type="operation", operator="*", value=152)

    # print(g.get_edge_data("A", "A"))

    # for (key, item) in g.get_edge_data("A", "A").items():
    #     print("v=", item)


    # DRAWING
    # pos = nx.spring_layout(g)

    # edge_labels = dict([((node_from, node_to), f'{data["type"]}')
    #                 for node_from, node_to, data in g.edges(data=True)])
    
    # nx.draw_networkx(g, pos, with_labels = True)

    # nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    # plt.show()

""" def sin(x):
    try:
        return x.__sin__()
    except AttributeError:
        return math.sin(x) """