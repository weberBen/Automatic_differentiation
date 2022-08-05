
import networkx as nx
import uuid
#%%

from .Operator import OPERATORS
from MathLib.FunctionReferences import FunctionRef

#%%



def _findOperation(computation_graph, operator: str, left, right):
    if operator not in OPERATORS:
        msg = "Invalid operator '" + str(operator) + "'"
        raise Exception(msg)
    
    if computation_graph.has_edge(left.id, right.id):

        for (_key, item) in  computation_graph.get_edge_data(left.id, right.id, default={}).items():
            if ("type" in item) and ("operator" in item):
                if item["type"]=="operation" and item["operator"]==operator:
                    return item["value"]
    
    if OPERATORS[operator]["commutative"]:

        if computation_graph.has_edge(right.id, left.id):
            for (_key, item) in  computation_graph.get_edge_data(right.id, left.id, default={}).items(): #not right and left id
                if ("type" in item) and ("operator" in item):
                    if item["type"]=="operation" and item["operator"]==operator:
                        return item["value"]
    
    return None

def _findFunction(computation_graph, func_name: str, node):

    if computation_graph.has_edge(node.id, func_name):
        for (_key, item) in  computation_graph.get_edge_data(node.id, func_name, default={}).items():
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

    if left.computation_graph is not right.computation_graph:
        left.computation_graph.add_edges_from(right.computation_graph.edges(data=True))
        left.computation_graph.add_nodes_from(right.computation_graph.nodes(data=True))

    computation_graph = left.computation_graph

    node_id = _findOperation(computation_graph, operator, left, right)
    if node_id is not None:
        new_node.id = node_id

        if right.computation_graph is not left.computation_graph:
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
    if node.requires_grad:
        computation_graph.add_node(node.id, value=value, type="variable", name=node.label)
    else:
        computation_graph.add_node(node.id, value=value, type="constant")


class Vector:

    def __init__(self, value, requires_grad=False, _id=None, _computation_graph=None, label=None):
        self.item = value
        self.id = _id if _id is not None else str(uuid.uuid4())

        self.computation_graph = _computation_graph if _computation_graph is not None else nx.MultiDiGraph()
        self.requires_grad = requires_grad
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

        output = Vector(function(self.item, *argv, **kwargs)) # computation in function are already detached from the computation graph since we pass only the value and not the vector
        _addFunction(self.computation_graph, func_name, self, output, argv, kwargs.items())

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
        G = nx.DiGraph()

        for (node, data) in self.computation_graph.nodes(data=True):
            if ("type" in data) and (data["type"]=="functionnal"):
                continue

            G.add_node(node, **data)
        
        for (node_from, node_to, data) in self.computation_graph.edges(data=True):
            if "type" in data :
                continue

            G.add_edge(node_from, node_to, **data)

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
        (computation_graph, ids_mapping) = self._getCleanComputationGraph(human_readable=False)

        nodes_to_process = [self.id]

        nodes_dv = {}
        nodes_dv[self.id] = 1 # dx/dx=1

        while len(nodes_to_process)>0:
            current_node_id = nodes_to_process.pop(0)
            
            #
            #
            #                    |----> children_node_1
            #                    |  
            #  parent_node_1 --->|---> children_node_2
            #                    |
            #                    |---->|---> current_node_id
            #                          |
            #  parent_node_2 --------->| ...
            # 
            # From the current node, we want to compute the derivative of the parent node, the predecessors
            # For that we need to get all the output relationships into which a parent node is implied, we need
            # to get the children of the parent node. One of thta children is the current node but a parent node could
            # have more than one output (so the current node could not the only child of that parent node) 

            node_appended = {}
            for parent_node_id in computation_graph.predecessors(current_node_id):
                parent_node = computation_graph.nodes[parent_node_id]
                dv_parent = 0 # dv=Derivative Value
                is_computable = True
                
                for child_node_id in computation_graph.successors(parent_node_id):
                    
                    # A parent node could be not computable at that time because we did not processed one of its child
                    # because that child is deeper in the graph
                    # Then we need to put process that child later but for the case when a parent node can be computed
                    # we use the same loop to compute the possible derivtaive

                    if child_node_id not in nodes_dv:
                        is_computable = False

                        if (child_node_id not in node_appended):
                            nodes_to_process.append(child_node_id)
                            node_appended[child_node_id] = True

                    
                    _node = computation_graph.nodes[child_node_id]

                    if child_node_id in nodes_dv:
                        _dv_child = nodes_dv[child_node_id] # total derivative for that child
                        _dv = None # derivative of the current child with respect to the parent
                        
                        if _node["type"]=="operator":
                            # The child node is part of an operation tha could be self referenced (reflective operator on the child node, ie x*x, x+x, x/x, ...)
                            # or it could be one node of that operation. The we need to get that second node because its value will be needed to compute the
                            # derivative of that child node (ie: if the child node is x and the other node is y, then to compute the derivative of x*y, we need the
                            # value of y)
                            # 
                            #
                            #                      other_node (y) --->|
                            #                                         |--->|--->  operation(+,-,*,/, ...) ----> another_node
                            #                                              |
                            #                |--->  child_1 (x) ---------->|
                            #                |
                            # parent_node -->|-----------------------------------------------> current_node
                            #                |
                            #                |--->  child_2 ---> ...
                            #
                            #
                            operator = _node["operator"]
                            
                            _node_operation_position = computation_graph.get_edge_data(parent_node_id, child_node_id)["operator_relative_position"]
                            
                            ##### retrieve the other node
                            _other_node = None
                            is_self_operation = False

                            _nodes_operation = list(computation_graph.predecessors(child_node_id)) # len in {1,2}
                            
                            if len(_nodes_operation)==1:
                                is_self_operation = True
                            else:
                                
                                _other_node_id = None
                                if _nodes_operation[0]!=parent_node_id:
                                    _other_node_id = _nodes_operation[0]
                                else:
                                    _other_node_id = _nodes_operation[1]
                            
                                _other_node = computation_graph.nodes[_other_node_id]
                            #####
                            
                            if operator=="+":
                                if is_self_operation: # x+x=2*x, dv=2
                                    _dv = 2
                                else: # x+y or y+x, dv=1
                                    _dv = 1
                            elif operator=="*":
                                if is_self_operation: # x^2, dv=2*x
                                    _dv = 2*_node["value"]
                                else: # x*y or y*x, dv=y
                                    _dv = _other_node["value"]
                            elif operator=="-":
                                if is_self_operation: # x-x=0, dv=0
                                    _dv = 0
                                else: #x-y or y-x, dv=1 or dv=-1
                                    if _node_operation_position=="left":
                                        _dv = 1
                                    else:
                                        _dv = -1
                            elif operator=="/":
                                if is_self_operation: # x/x=1, dv=0
                                    _dv=0
                                else: # x/y or y/x, dv=1/y or dv=-y/x^2
                                    if _node_operation_position=="left":
                                        _dv = 1/_other_node["value"]
                                    else:
                                        _dv = -1*_other_node["value"]/(_node["value"]**2)
                            else:
                                msg = "invalid operator '{0}'".format(operator)
                                raise Exception(msg)
                        
                        elif _node["type"]=="function":

                            func_name = _node["func_name"]
                            func_argv = _node["func_argv"]
                            func_kwargs = dict(_node["func_kwargs"])
                            input_value = parent_node["value"] # or input_value = _nodes["input_value"]

                            _dv = FunctionRef.getNew(func_name)._derivative(input_value, *func_argv, **func_kwargs)
                        else:
                            msg = "invalid node type '{0}'".format(_node["type"])
                            raise Exception(msg)
                        
                        dv_parent += _dv_child*_dv
                        # Todo: optimise the computation by storing the partial sum for the rest of the computation later
                
                if is_computable:
                    nodes_dv[parent_node_id] = dv_parent
                    
        return nodes_dv

#%%
