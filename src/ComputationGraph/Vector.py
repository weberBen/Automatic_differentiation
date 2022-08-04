
import networkx as nx
import uuid
#%%
import MathLib
from Operator import OPERATORS

#%%



def _findOperation(computation_graph, operator: str, left, right):
    if operator not in OPERATORS:
        msg = "Invalid operator '" + str(operator) + "'"
        raise Exception(msg)
    
    if computation_graph.has_edge(left.id, right.id):
        data = computation_graph.get_edge_data(left.id, right.id, default={})

        #for (key, item) in data.items():
        item = data
        if ("type" in item) and ("operator" in item):
            if item["type"]=="operation" and item["operator"]==operator:
                return item["value"]
    
    if OPERATORS[operator]["commutative"]:

        if computation_graph.has_edge(right.id, left.id):
            data = computation_graph.get_edge_data(right.id, left.id, default={}) #not right and left id
            #for (key, item) in data.items():
            item = data
            if ("type" in item) and ("operator" in item):
                if item["type"]=="operation" and item["operator"]==operator:
                    return item["value"]
    
    return None

def _findFunction(computation_graph, func_name: str, node):

    if computation_graph.has_edge(node.id, func_name):
        data = computation_graph.get_edge_data(node.id, func_name, default={})

        #for (key, item) in data.items():
        item = data
        if ("type" in item) and (item["type"] == "function"):
            return item["value"]
    
    return None

def _addFunction(computation_graph, func_name: str, node: object, new_node: object, argv, kwargs):
    
    node_id = _findFunction(computation_graph, func_name, node)
    if node_id is not None:
        new_node.id = node_id
        new_node.computation_graph = computation_graph

        return

    computation_graph.add_node(new_node.id, func_name=func_name, type="function", value=new_node.item, input_value=node.item, func_argv=argv, func_kwargs=kwargs)

    if not computation_graph.has_node(func_name):
        computation_graph.add_node(func_name, value=None, type="functionnal")
    
    computation_graph.add_edge(node.id, new_node.id)
    
    computation_graph.add_edge(node.id, func_name, type="function", func_name=func_name, value=new_node.id)

    
    new_node.computation_graph = computation_graph
    
def _addOperation(computation_graph, operator: str, left: object, right: object, new_node):
    
    if operator not in OPERATORS:
        msg = "Invalid operator '" + str(operator) + "'"
        raise Exception(msg)

    is_constant = False
    if right.computation_graph.size()==0: # if reuiqred_autograd then computation_graph is at least populated with a single node that represente the vector
        is_constant=True
    

    left.computation_graph.add_edges_from(right.computation_graph.edges(data=True))
    left.computation_graph.add_nodes_from(right.computation_graph.nodes(data=True))

    computation_graph = left.computation_graph

    node_id = _findOperation(computation_graph, operator, left, right)
    if node_id is not None:
        new_node.id = node_id

        right.computation_graph.add_edges_from(computation_graph.edges(data=True))
        right.computation_graph.add_nodes_from(computation_graph.nodes(data=True))

        new_node.computation_graph = computation_graph

        return

    computation_graph.add_node(new_node.id, operator=operator, type="operator", value=new_node.item)
    
    computation_graph.add_edge(left.id, new_node.id, operator_relative_position="left")

    if right.id!=left.id:
        if not computation_graph.has_node(right.id):
            computation_graph.add_node(right.id, type=("constant" if is_constant else ""), value=(right.item if is_constant else None))
        
        computation_graph.add_edge(right.id, new_node.id, operator_relative_position="right")
    
    computation_graph.add_edge(left.id, right.id, type="operation", operator=operator, value=new_node.id)

    # Partage le même object puisque dès le depart on va avoir à la base des variables 
    # et ont va set le même obj pour ces variables, les nodes left et right auront les mêmes obj
    # donc la suite des opération se fera sur le même obj partagé par tous les noeuds et en partiuclier
    # les variables
    # cela permet en autre de partager toutes les opérations mêmes si elles sont faites de manière indépendnate
    # par example sin(x*y) + sin(x*y) va donner le calcul de x*y dans le premier sin puis le x*y dans le deuxième
    # et seulement ensuite le calcul de sin pour ces deux x*y
    # comme sin(x*y) sera ajouté à l'obj partagé, on aura enregistré l'opération dans x et y donc le 2ème calcul
    # de sin(x*y) pourra être détécté
    # si on fait juste computation_graph = left.computation_graph puis ensuite à la fin seuelment
    # right.computation_graph.add_nodes_from(computation_graph.nodes(data=True)) et right.computation_graph.add_edges_from(computation_graph.edges(data=True))
    # on a 2 obj séparé pour x et y donc lorsque les calculs vont se faire de manière indépendante on ne pourra plus les detecter
    # ex: le premier sin(x*y) va se faire sur l'arbre de gauche qui ne sera pas repercuté sur l'arbre des variables de base
    # puisque dès la base on a 2 objs différent, donc le 2ème sin(x*y) va donner un calcul à part entière
    right.computation_graph = computation_graph

    new_node.computation_graph = computation_graph

def _initComputationGraph(computation_graph, node, value):
    if node.required_autograd:
        computation_graph.add_node(node.id, value=value, type="variable", name=node.label)
    else:
        computation_graph.add_node(node.id, value=value, type="constant")


class Vector:

    def __init__(self, value, required_autograd=False, _id=None, _computation_graph=None, label=None):
        self.item = value
        self.id = _id if _id is not None else str(uuid.uuid4())

        self.computation_graph = _computation_graph if _computation_graph is not None else nx.DiGraph()
        self.required_autograd = required_autograd
        self.label = label

        _initComputationGraph(self.computation_graph, self, value=self.item)

        
    
    def item(self):
        return self.item   
    
    def __del__(self):#garbage collector
        pass

    def __add__(self, v):
        _operator = "+"

        if type(v)==int or type(v)==float:
            v = Vector(v)
        
        if type(v) is Vector:

            output = Vector(self.item+v.item)

            _addOperation(self.computation_graph, _operator, self, v, output)
            
            return output
        
        return None
    
    def __mul__(self, v):
        _operator = "*"

        if type(v)==int or type(v)==float:
            v = Vector(v)
        
        if type(v) is Vector:

            output = Vector(self.item*v.item)
            _addOperation(self.computation_graph, _operator, self, v, output)
            return output
        
        return None
    
    def __sub__(self, v):
        _operator = "-"

        if type(v)==int or type(v)==float:
            v = Vector(v)
        
        if type(v) is Vector:

            output = Vector(self.item-v.item)
            _addOperation(self.computation_graph, _operator, self, v, output)
            return output
        
        return None
    
    def __truediv__(self, v):
        _operator = "/"

        if type(v)==int or type(v)==float:
            v = Vector(v)
        
        if type(v) is Vector:

            output = Vector(self.item/v.item)
            _addOperation(self.computation_graph, _operator, self, v, output)
            return output
        
        return None
    
    def __apply__(self, func_name, function, *argv, **kwargs):

        output = Vector(function(self.item))
        _addFunction(self.computation_graph, func_name, self, output, argv, kwargs.items())

        return output

    def __sin__(self):
        func_name = "sin"

        output = Vector(MathLib.sin(self.item))
        _addFunction(self.computation_graph, func_name, self, output)
        return output
    
    def __cos__(self):
        func_name = "cos"

        output = Vector(MathLib.cos(self.item))
        _addFunction(self.computation_graph, func_name, self, output)
        return output
    
    def __rmul__(self, v):
        return Vector(v)*self
    
    def __rsub__(self, v):
        return Vector(v)-self
    
    def __rtruediv__(self, v):
        return Vector(v)/self
    
    def __radd__(self, v):
        return Vector(v)+self
    
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return self
    
    def __str__(self):
        return "Vector({0})".format(self.item)
    
    def _getCleanComputationGraph(self, human_readable=False):
        G = nx.Graph.copy(self.computation_graph)

        edges_to_remove = [(node_from, node_to) for (node_from, node_to, data) in G.edges(data=True) if ("type" in data)]
        nodes_to_remove = [node for (node, data) in G.nodes(data=True) if (("type" in data) and (data["type"]=="functionnal"))]

        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(nodes_to_remove)

        mapping = None
        if human_readable:
            mapping = {}

            sorted_nodes = sorted(G.nodes(data=False))
            for index in range(len(sorted_nodes)):
                node_id = sorted_nodes[index]
                mapping[node_id] = index

            G = nx.relabel_nodes(G, mapping)

        return (G, mapping)
    
    def backward(self):
        computation_graph = self._getCleanComputationGraph()

        root = self.computation_graph.nodes[self.id]
        node_id = self.id

        node_values = {}

        node_values[self.id] = 1 # derivate of v WRT to v is always 1

        while True:
            for parent_node_id in computation_graph.successors(node_id):
                parent_node = computation_graph.nodes[parent_node_id]

                child_v = node_values[node_id]
                dv = None

                if parent_node["type"]=="operator":
                    operator = parent_node["operator"]
                    items = list(computation_graph.predecessors(parent_node_id))


                    items = list(computation_graph.predecessors(parent_node_id))
                    node_id_1 = items[0]
                    node_id_2 = None

                    if len(items)==1: #power of 2
                        node_id_2 = node_id_1
                    else:
                        node_id_2 = items[1]

                    current_node_id = None
                    other_node_id = None
                    if node_id_1 == parent_node_id:
                        current_node_id = node_id_1
                        other_node_id = node_id_2
                    else:
                        current_node_id = node_id_2
                        other_node_id = node_id_1
                    
                    current_edge = None
                    other_edge = None
                    edge_1 = None
                    edge_2 = None

                    if computation_graph.has_edge(parent_node_id, node_id_1):
                        edge_1 = computation_graph.get_edge_data(parent_node_id, node_id_1)

                    else:
                        edge_1 = computation_graph.get_edge_data(node_id_1, parent_node_id)
                    
                    if computation_graph.has_edge(parent_node_id, node_id_2):
                        edge_2 = computation_graph.get_edge_data(parent_node_id, node_id_2)
                    else:
                        edge_2 = computation_graph.get_edge_data(node_id_2, parent_node_id)

                    if current_node_id==node_id_1:
                        current_edge = edge_1
                        other_edge = edge_2
                    else:
                        current_edge = edge_2
                        other_edge = edge_1
                    
                    operator_relative_position = current_edge["operator_relative_position"]

                    if operator==OPERATORS["+"]:
                        dv = 1
                    elif operator==OPERATORS["-"]:
                        if operator_relative_position=="left":
                            dv = 1
                        else:
                            dv = -1
                    elif operator==OPERATORS["*"]:
                        dv = computation_graph.nodes[other_node_id]["value"]
                    elif operator==OPERATORS["/"]:
                        if operator_relative_position=="left":
                            dv = 1/computation_graph.nodes[other_node_id]["value"]
                        else:
                            dv = -1/(parent_node["value"]**2)
                    
                    node_values[parent_node_id] = child_v*dv
                
                elif parent_node["type"]=="function":
                    pass


#%%
