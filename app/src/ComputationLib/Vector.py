
import networkx as nx
import uuid

from regex import R
#%%

from .Operator import OPERATORS
from MathLib.FunctionReferences import FunctionRef

#%%

def _findOperation(computation_graph, operator: str, left, right):
    if operator not in OPERATORS:
        msg = "Invalid operator '" + str(operator) + "'"
        raise Exception(msg)
    
    if computation_graph.has_edge(left.id, right.id):

        for (_key, item) in  computation_graph.get_edge_data(left.id, right.id, default={}).items():  # multi edges graph
            if ("type" in item) and ("operator" in item):
                if item["type"]=="operation" and item["operator"]==operator:
                    return item["value"]
    
    if OPERATORS[operator]["commutative"]:

        if computation_graph.has_edge(right.id, left.id):
            for (_key, item) in  computation_graph.get_edge_data(right.id, left.id, default={}).items():  # multi edges graph
                if ("type" in item) and ("operator" in item):
                    if item["type"]=="operation" and item["operator"]==operator:
                        return item["value"]
    
    return None


def _addOperation(operator: str, left, right, output):
    """
        Register an operation (+,-, ...) to the computation graph

        Parameters
        ----------
            operator : string
                name of the operator
            left : Vector
                left sided member of the operation (left_obj operator right_obj)
            right : Vector
                right sided member of the operation (left_obj operator right_obj)
            output : Vector
                result of the operation wrapped around a Vector
    """

    if operator not in OPERATORS:
        msg = "Invalid operator '" + str(operator) + "'"
        raise Exception(msg)

    # The computation need to be shared by all the vectors that has been implied into an operation
    # For example : 
    #   z = (x+y) + sin(x+y) - 3
    # The operation (x+y) must correspond to an unique node inside the computation graph and not a node for (x+y) and for the sinus of (x+y)
    # In other words we need to have :
    #
    #   (x)---->|-->|---> (+) --->|---> (sin)--->|
    #               |             |              |
    #   (y)-------->|             |              |----->|---> (+) ---------->|
    #                             |                     |                    |                   
    #                             |-------------------->|                    |------>|---> (-)
    #                                                                                |
    #   (3)--------------------------------------------------------------------------|
    #
    # Instead of :
    #
    #   (x)-->|----------------------------------->|
    #         |                                    |
    #         |                  (y)------->|----->|------> (+)----------> (sin)------>|
    #         |                             |                                          |----->|----> (+)------>|
    #         |                             |--------------->|------------->(+)-------------->|                |
    #         |                                              |                                                 |
    #         |--------------------------------------------->|                                                 |-------->|-----> (-)
    #                                                                                                                    |
    #   (3)------------------------------------------------------------------------------------------------------------->|
    #
    # But in fact the Interpreter of a langage generally make the calculation of the inner object of a function and then the rest
    # It means that the (x+y) inside the sinus function will be computed first and if the computation graph of y is not updated for each operation/function
    # Then by the time the interpreter will go to the first (x+y) the computation graph of y will be blank and it will recreate another node
    # To sync the computation graph for each operation we make sure that the computation graph of each member (left and right) reference the same 
    # (not a copy but the same object)
    # All vector that has been related in a operation share the same computation graph object
    # When an operation occurs we start to merge the computation graph of the two members and then erase the old computations graph to the same reference
    # In particular, at the begining when two variable are related in an operation their nearly empty computation graph will be replace by the same object as
    # reference, then later when one of that variable will be implied in another operation the computation graph of that variable will be immediatly sync since
    # it's the same object
    # For example :
    #   x,y variable
    #   d = x+y
    #   z = y*2 + 4
    #   w = x*y
    # At w the computation graph of y will include all the operations made on x and y until now
    # In fact the computation graphs of x,y,d,z are the same object
    # Then if j is a variable
    #   b = j+y
    # After b the computation graph of j will be the one from x,y,d,z and now b
    #
    # We do not register constants, but we could. In other words the operation (x+y)*3 + (x+y)*3 will create a unique node for (x+y) but not for (x+y)*3
    # Two other nodes that represent the operator multiplication will be created involving the node (x+y) and two new nodes fo the constant 3
    #
    #   (x)-------->|
    #               |----->|-----> (+)---->|---------->|
    #   (y)--------------->|               |           |
    #                                      |           |--------->|------> (*)------>|
    #                                      |   (3)--------------->|                  |------->|---------> (+)
    #                                      |                                                  |
    #                                      |--------------->|------------> (*)--------------->|
    #                                                       |
    #                                (3)------------------->|    
    #

    if left.computation_graph is not right.computation_graph:
        # If the two objects are the same then adding nodes, edges to left will change the size of the graph of the righ
        # That will produce iteration on changing size list
        left.computation_graph.add_edges_from(right.computation_graph.edges(data=True))
        left.computation_graph.add_nodes_from(right.computation_graph.nodes(data=True))

    # for ease of use (could be computation graph of the rigth)
    computation_graph = left.computation_graph

    node_id = _findOperation(computation_graph, operator, left, right)
    if node_id is not None:
        # The operation already exists in the computation graph
        # Then we set the id out the output vector to the one register inside the computation graph (since it need
        # to represents the same vector)
        output.id = node_id

        if right.computation_graph is not left.computation_graph:
            right.computation_graph.add_edges_from(computation_graph.edges(data=True))
            right.computation_graph.add_nodes_from(computation_graph.nodes(data=True))

        # sync the computation graph of the ouput vector
        output.computation_graph = computation_graph

        return

    # node that represents the operation
    computation_graph.add_node(output.id, operator=operator, type="operator", value=output.item)
    
    # left edge of the operation
    computation_graph.add_edge(left.id, output.id, operator_relative_position="left")

    if right.id!=left.id:
        # reflective operation (x+x, x*x, x/x) will produce two identique edge that we need to avoid

        # right edge of the operation
        computation_graph.add_edge(right.id, output.id, operator_relative_position="right")
    
    # Help to retrieve the operation between two nodes
    # Else we would have to check all the children of the left node, then find a match for the right node 
    # and then iterate over each possibility to check if the shared node is an operator and match desired operator
    # That edge will be removed once the computation graph is fully built (by Vector#_getCleanComputationGraph)
    computation_graph.add_edge(left.id, right.id, type="operation", operator=operator, value=output.id)

    # sync the computation graph of the right member
    right.computation_graph = computation_graph
    # sync the computation graph of the output vector
    output.computation_graph = computation_graph

#%%

def _findFunction(computation_graph, func_name: str, node):

    if computation_graph.has_edge(node.id, func_name):
        for (_key, item) in  computation_graph.get_edge_data(node.id, func_name, default={}).items(): # multi edges graph
            if ("type" in item) and (item["type"] == "function"):
                return item["value"]
    
    return None

def _addFunction(func_name: str, node: object, output: object, argv, kwargs):
    computation_graph = node.computation_graph

    node_id = _findFunction(computation_graph, func_name, node)
    if node_id is not None:
        output.id = node_id
        output.computation_graph = computation_graph

        return

    computation_graph.add_node(output.id, func_name=func_name, type="function", value=output.item, input_value=node.item, func_argv=argv, func_kwargs=kwargs)
    
    computation_graph.add_edge(node.id, output.id)

    if not computation_graph.has_node(func_name):
        # Help to retrieve the operation between two nodes
        # That node will be removed once the computation graph is fully built (by Vector#_getCleanComputationGraph)
        computation_graph.add_node(func_name, value=None, type="functionnal")

    
    # Help to retrieve the operation between two nodes
    # That edge will be removed once the computation graph is fully built (by Vector#_getCleanComputationGraph)
    computation_graph.add_edge(node.id, func_name, type="function", func_name=func_name, value=output.id)


    output.computation_graph = computation_graph
    
#%%

def _initComputationGraph(computation_graph, node, value):
    """
        Initialize the computation graph of each vector
        Each vector (variable or constant) has a computation graph with at least one node inside

        Parameters
        ----------
            computation_graph : netwrokx
                empty graph
            node : Vector instance

            value : int/float
                value of the vector
    """
    # Vector can either be a variable or a constant
    if node.requires_grad:
        computation_graph.add_node(node.id, value=value, type="variable", name=node.label, ref=node) 
        # ref is used to access the node later to set the gradient attriobute for that variable
    else:
        computation_graph.add_node(node.id, value=value, type="constant")

#%%

class Vector:
    """
        Tensor implementation that deal with autograd
    """

    def __init__(self, value, requires_grad=False, label=None, _id=None, _computation_graph=None, _init_computation_graph=True):
        """
        Parameters
        ----------
            value : int/float
                value of the vector
            requires_grad : boolean (optionnal)
                true if the current vector need to be handled as a variable, else false
            label : boolean (optionnal)
                indicates if we need to display that label for the corresponding node in the computation graph when plotting it
                (purely aesthetic option)
        
        Internal Parameters (do not use manually)
        ----------
            _id : string (optionnal)
                id of the vector that is used by the computation graph to keep track of the operation
            _computation_graph : networkx graph (optionnal)
                use to set the computation graph of the new vector created by operation (+,-, sin, cos, ...)
            _init_computation_graph: boolean (optionnal)
                initiliazed of not the computation graph. Parameters ignored if the vector requires autograd
        """

        _init_computation_graph = _init_computation_graph if not requires_grad else True


        self.item = value
        self.id = _id if _id is not None else str(uuid.uuid4())
        
        
        if not _init_computation_graph:
            self.computation_graph = None
        else:
            # Each vector starts with an empty graph that will hold all operations the vector is being implied into
            self.computation_graph = _computation_graph if _computation_graph is not None else nx.MultiDiGraph()

            # The computation graph is multi edges directed graph
            # To build the computation graph we need to add edge between nodes that help to retrieve the operation/function
            # already made
            # For example for the operation x+y, we will have :
            #   (x)-------->|
            #               |----->|-----> (+)
            #   (y)--------------->|
            #
            # But also an edge between (x) and (y) that will hold the information that there is an addition between (x) and (y)
            # That edge will be later removed when the computation graph will be fully created
            # Since there could be multiple operation/function between to objects we need a graph with multiple edges
            # Denotes that the edges that will remain in the final computation graph like (x, +) and (y, +) are unique (there could not be multplie edges
            # between the nodes (x) and  (+), since a node is either a variable, constant, an operator (that takes only two nodes as input), or function
            # (that takes one node as input))
            # Here the node (+) is just for presentation purposes but in the real computation graph the id of a node is not the operator itself
            # `(random_id) = +` is a more correct representation for that node. In other words, there is not a unique node for each addition operation
            # 
            # The finaly computation graph is directed graph with single edge between two nodes
        
        self.requires_grad = requires_grad

        # optionnal value for displaying label when plotting the graph
        self.label = label

        # partial derivative value that will be updated when computing the gradient of the function with respect to that variable
        self.grad = 0
        
        if _init_computation_graph:
            _initComputationGraph(self.computation_graph, self, value=self.item)

    
    def item(self):
        return self.item

    def __operation__(self, operator, v):
        """
        Compute operation (+,-, ...) for the vector

        Parameters
        ----------
            operator : string
                name of the operation
            v : object
                Object that is implied into the operation (ie: current_vector operator v)
        """
        if type(v)==int or type(v)==float:
            v = Vector(v)
        
        if type(v) is Vector:

            value = None
            if operator=="+":
                value = self.item+v.item
            elif operator=="-":
                value = self.item-v.item
            elif operator=="*":
                value = self.item*v.item
            elif operator=="/":
                value = self.item/v.item
            else:
                msg = "Invalid operator '{0}' to apply".format(operator)
                raise Exception(msg)
            
            output = Vector(value, _init_computation_graph=False)
            # computation graph of the output will be erease at the end of the operation to share the one from left and right member of the operation

            # register the operation in the computation graph
            _addOperation(operator, self, v, output)
            
            return output
        
        return None

    def __add__(self, v):
        return self.__operation__("+", v)
      
    def __mul__(self, v):
        return self.__operation__("*", v)
    
    def __sub__(self, v):
        return self.__operation__("-", v)
    
    def __truediv__(self, v):
        return self.__operation__("/", v)
    
    def __apply__(self, func_name, computation_function, *argv, **kwargs):
        """
        Apply a function (sin, cos, ...) to the current vector

        Parameters
        ----------
            func_name : string
                name of the function (unique)
            computation_function : function
                Function that handle the computation on real value (int/float)
            argv : tuples
                unnamed parameters of the computation_function function
            kwargs: dict
                named parameters of the computation_function function
        """

        # computation inside the function are already detached from the computation graph since we pass only the value (int/float) and not the vector
        output = Vector(computation_function(self.item, *argv, **kwargs), _init_computation_graph=False)

        # register the function in the computation graph
        _addFunction(func_name, self, output, argv, kwargs.items())

        return output
    
    def __rmul__(self, v):
        # Go to __mul___ case by converting the value to vector
        return Vector(v)*self
    
    def __rsub__(self, v):
         # Go to __sub___ case by converting the value to vector
        return Vector(v)-self
    
    def __rtruediv__(self, v):
         # Go to __truediv__ case by converting the value to vector
        return Vector(v)/self
    
    def __radd__(self, v):
        # Go to __add__ case by converting the value to vector
        return Vector(v)+self
    
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return self
    
    def __str__(self):
        return "Vector({0})".format(self.item)
    
    def _getCleanComputationGraph(self, human_readable=False):
        """
        Remove edges and nodes use to built the computation graph among the final edges and nodes that are needed

        Parameters
        ----------
            human_readable : boolean (optionnal)
                convert the id to interger that start to 0
        
        Return
        ----------
            computation_graph :  networkx graph
                Computation graph reduces to only the nodes and edges needed for gradient computation
            mapping : dict/None
                if human_readable is false then return None
                Else return the mapping between the initial node ids and the human readable ids that are used by the computation graph

                Until the nodes of the computation graph change, the mapping will be the same since the nodes are being sorted by id before
                building the mapping
        """

        G = nx.DiGraph()

        
        for (node_id, data) in self.computation_graph.nodes(data=True):
            if ("type" in data) and (data["type"]=="functionnal"):
                continue
            
            G.add_node(node_id, **data)
        
        for (node_from, node_to, data) in self.computation_graph.edges(data=True):
            if "type" in data :
                continue

            G.add_edge(node_from, node_to, **data)
        
        ##### remove unused node

        # An unsed node is a node that as no outputs/no sucessors (the current node where the gradient will be computed is excluded)
        # An unsed node appears when a variable is part of a calculation that is not reused by the final vector where the gradient is being computed
        # For example :
        #   x, y, z vector variables
        #   d = x+y
        #   p = x*3
        #   g = y + z + p
        #   gradient of g
        #
        # Here the node built from d (with x+y) is unused because not reused in the final object g where the gradient need to be computed
        # But, since the computation graph is shared across all variables, at g the operation (x+y) will already be registered in the computation graph of y
        # Then it will be part of the computation graph of g
        # To ease the computation it's better to remove theses nodes
        
        unused_nodes = {}
        for node_id in G.nodes():
            if node_id in unused_nodes:
                continue
            try:
                next(G.successors(node_id))
            except StopIteration: # node has no output/child (then unused node)

                if node_id!=self.id:
                    unused_nodes[node_id] = True

                    nodes_to_proceed = [node_id]

                    # We need to track back all the nodes related to an unsued node
                    # because removing a node could reveal other unused ndoes
                    while len(nodes_to_proceed)>0:
                        _node_id = nodes_to_proceed.pop(0)

                        for pred_node_id in G.predecessors(_node_id):
                            
                            count_used = 0
                            count_unused = 0
                            for succ_node_id in G.successors(pred_node_id):

                                if succ_node_id in unused_nodes:
                                    count_unused+=1
                                else:
                                    count_used+=1
                            
                            if (count_unused+count_used)==count_unused:
                                
                                unused_nodes[pred_node_id] = True
                                nodes_to_proceed.append(pred_node_id)
        
        
        # start to removing the unused edges produced by the identification of the unused nodes
        for (node_from, node_to) in list(G.edges()):  
            if (node_to in unused_nodes):
                G.remove_edge(node_from, node_to)
        
        # finally remove the unused nodes
        for (node_id, val) in unused_nodes.items():
            G.remove_node(node_id)
        
        #####

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
        
        # This list will hold a serie of nodes id that satisfay the dependencies of each node to allow the computation
        # This is not necessary the shortest path to resolve the problem
        nodes_to_process = [self.id] 

        nodes_dv = {}
        nodes_dv[self.id] = 1 # dx/dx=1

        # node to perform autograd onto is the variable (x=variable, z=x, autograd on z)
        root_node = computation_graph.nodes[self.id]
        if ("type" in root_node) and (root_node["type"]=="variable") and ("ref" in root_node):
            variable_ref_obj = root_node["ref"]
            variable_ref_obj.grad = 1
            
            return nodes_dv
        
        while len(nodes_to_process)>0:
            current_node_id = nodes_to_process.pop(0)

            nodes_visited = {}
            
            #
            #
            #                    |----> child_node_1
            #                    |  
            #  parent_node_1 --->|----> child_node_2
            #                    |
            #                    |---->|---> current_node_id
            #                          |
            #  parent_node_2 --------->| ...
            # 
            # From the current node, we want to compute the derivative of the parent node, the predecessors
            # For that we need to get all the output relationships into which a parent node is implied, we need
            # to get the children of the parent node. One of thta children is the current node but a parent node could
            # have more than one output (so the current node could not the only child of that parent node) 
            
            for parent_node_id in computation_graph.predecessors(current_node_id): # avoid using list that will load all nodes in memory, instead use cursor

                parent_node = computation_graph.nodes[parent_node_id]

                if ("type" in parent_node) and (parent_node["type"]=="constant"): # else will use the chain rules to compute derivative as if it was a variable
                    continue
                
                ##### compute the derivative value of the parent node with all its children derivatives value if already computed
                dv_parent = 0 # dv=Derivative Value
                is_computable = True

                for child_node_id in computation_graph.successors(parent_node_id):

                    parent_node = computation_graph.nodes[parent_node_id]

                    # A parent node could be not computable at that time because we did not processed one of its child
                    # because that child is deeper in the graph

                    if child_node_id not in nodes_dv:
                        is_computable = False

                        if (child_node_id not in nodes_visited):
                            nodes_to_process.append(child_node_id)
                            nodes_visited[child_node_id] = True

                    
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
                            #                                         |--->|--->   child_1 = operation(+,-,*,/, ...) ----> another_node
                            #                                              |
                            #                    |------------------------>|
                            #                    |
                            # parent_node (x) -->|-----------------------------------------------> current_node
                            #                    |
                            #                    |--->  child_2 ---> ...
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
                                    _dv = 2*parent_node["value"]
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
                                        _dv = -1*_other_node["value"]/(parent_node["value"]**2)
                            else:
                                msg = "invalid operator '{0}'".format(operator)
                                raise Exception(msg)
                        
                        elif _node["type"]=="function":
                            
                            #
                            #                    |---> child_1 = function(sin, cos, log, ...) ----> another_node
                            #                    |
                            # parent_node ------>|----------------------------------------------> current_node
                            #                    |
                            #                    |--->  child_2 ---> ...
                            #
                            #

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
                
                ##### end computing parent node derivative value

                if is_computable: 
                    # all the children derivative value has been previously computed
                    # The parent derivative which is the mainly consist of the sum of the children derivative value, then can be computed
                    nodes_dv[parent_node_id] = dv_parent

                    if ("type" in parent_node) and (parent_node["type"]=="variable") and ("ref" in parent_node):
                        # retrieve the variable vector object to update its grad attribute with the appropriate value
                        variable_ref_obj = parent_node["ref"]
                        variable_ref_obj.grad = dv_parent

                if is_computable and (parent_node_id not in nodes_visited):
                    # The parent node will act as a child for other nodes with a complete derivative value computed
                    # Then we can check if the derivatives of its predecessors can be computed (as that parent node which will be a child
                    # node for its predecessors has a derivative value). Else when the parent node will be proceeded we will had the needed children nodes
                    # to the list
                    nodes_to_process.append(parent_node_id)
                    nodes_visited[parent_node_id] = True

        
        return nodes_dv

#%%
