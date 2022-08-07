import networkx as nx
from pyvis.network import Network
#%%
from .Operator import OPERATORS
from .Vector import Vector
#%%

class ComputationGraphProcessor:
    def __init__(self, end_vector: Vector, human_readable=True):
        """
        Parameters
        ----------
            end_vector : Vector
                vector to perform actions on
            human_readable: boolean (optionnal)
                rename each node id to integer starting from 0
        """

        self.end_vector = end_vector
        (self.computation_graph, self.graph_mapping) = self.end_vector._getCleanComputationGraph(human_readable=human_readable)
    
    @staticmethod
    def _getVariableNodeName(node_id, data, prefix=None):

        prefix_named_var = None
        prefx_unnamed_var = None
        if type(prefix) is tuple:
            (prefix_named_var, prefx_unnamed_var) = prefix
        else:
            prefix_named_var = prefix
            prefx_unnamed_var = prefix
        
        if ("name" in data) and (data["name"]):
            return str((prefix_named_var if prefix_named_var else "") + str(data["name"]))
        else:
            return str((prefx_unnamed_var if prefx_unnamed_var else "") + str(node_id))

    @staticmethod
    def _formatNodeLabel(node_id, data):
        if "type" not in data:
            if "value" in data:
                return str(data["value"])
            return ""
        
        if data["type"]=="operator":
            return data["operator"]
        
        if data["type"]=="constant":
            return f'const({data["value"]})'
        
        if data["type"]=="variable":
            var_name = ComputationGraphProcessor._getVariableNodeName(node_id, data, ("", "var_"))
            return f'var({var_name})={data["value"]}'
        
        if data["type"]=="function":
            return data["func_name"]
        
        return str(data["type"])
    
    @staticmethod
    def _formatNodeTitle(node_id, data, display_nodes_value):
        title = f'id({node_id})'
        if display_nodes_value and ("value" in data):
            title =  title + '''
            value(''' + str(data["value"]) + ''')
            '''
        
        return title

    def _getLocalNodeId(self, node_id):
        if self.graph_mapping is None:
            return node_id
        return self.graph_mapping[node_id]
    
    def draw(self, layout=False, width="100%", height="100%", display_nodes_value=False, filename='nx.html'):
        """
        Draw computation graph

        Parameters
        ----------
            layout : boolean (optionnal)
                pyvis layout option
            width: string (optionnal)
                pyvis width option
            height: string (optionnal)
                pyvis height option
            display_nodes_value: boolean (optionnal)
                display computed value of each node in the computation graph
            filename: string (boolean)
                file to write the html (pyvis) drawing of the graph
        """
        G = self.computation_graph

        nt = Network(width, height, directed=True, layout=layout)
        for (node_id, data) in G.nodes(data=True):
            font_size = "15"

            if ("type" in data) and (data["type"]=="operator"):
                font_size = "25"
            
            label = ComputationGraphProcessor._formatNodeLabel(node_id, data)
            title = ComputationGraphProcessor._formatNodeTitle(node_id, data, display_nodes_value)
            
            nt.add_node(node_id, label=label, group=data["type"], title=title, font=f'{font_size}px')
        
        for (node_from_id, node_to_id, data) in G.edges(data=True):

            label = None
            if "operator_relative_position" in data:
                label = data["operator_relative_position"]
            
            nt.add_edge(node_from_id, node_to_id, physics=True, title=label)
        

        root_node_id = self._getLocalNodeId(self.end_vector.id)

        nt.add_node("output", label="output", group="output", title=f'id({root_node_id})')
        nt.add_edge(root_node_id, "output", physics=True, title="output")

        nt.show(filename)

    def rebuildExpression(self, track_origin=False):

        computation_graph = self.computation_graph

        leafs = []
        root_node_id = self._getLocalNodeId(self.end_vector.id)
        graph_root = None # graph root could be different from the end vector we want to evaluate the expression
        var_count = 0
        nodes_eval = {}

        for (node_id, data) in computation_graph.nodes(data=True):
            try:
                next(computation_graph.predecessors(node_id))
            except StopIteration:
                if "type" in data: 
                    if data["type"]=="variable":
                        leafs.append(node_id)

                        var_name = ComputationGraphProcessor._getVariableNodeName(node_id, data, ("", "var_"))
                        nodes_eval[node_id] = var_name

                        var_count += 1
                    
                    elif data["type"]=="constant":
                        leafs.append(node_id)
                        nodes_eval[node_id] = f'{data["value"]}'
                
            try:
                next(computation_graph.successors(node_id))
            except StopIteration:
                graph_root = node_id
        
        processing = leafs

        while len(processing)>0:

            new_processing = []
            for node_id in processing:
                for parent_node_id in computation_graph.successors(node_id):
                    parent_node = computation_graph.nodes[parent_node_id]

                    if parent_node["type"]=="operator":
                        operator = parent_node["operator"]
                        
                        items = list(computation_graph.predecessors(parent_node_id))
                        node_id_1 = items[0]
                        node_id_2 = None

                        if len(items)==1: #power of 2
                            node_id_2 = node_id_1
                        else:
                            node_id_2 = items[1]

                        if (node_id_1 not in nodes_eval) or (node_id_2 not in nodes_eval):
                            new_processing.append(node_id)
                            continue

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

                        
                        operator_commutative = OPERATORS[operator]["commutative"]

                        eval_node_1 = None
                        eval_node_2 = None

                        if operator_commutative:
                            tmp = sorted([node_id_1, node_id_2]) # to have always the same order if the operation is reused later
                            eval_node_1 = nodes_eval[tmp[0]]
                            eval_node_2 = nodes_eval[tmp[1]]
                        else:
                            if edge_1["operator_relative_position"]=="left":
                                eval_node_1 = nodes_eval[node_id_1]
                                eval_node_2 = nodes_eval[node_id_2]
                            else:
                                eval_node_1 = nodes_eval[node_id_2]
                                eval_node_2 = nodes_eval[node_id_1]


                        eval_str = None
                        if track_origin:
                            eval_str = f'({eval_node_1} {operator} {eval_node_2})@{parent_node_id}'
                        else:
                            eval_str = f'({eval_node_1} {operator} {eval_node_2})'
                        
                        nodes_eval[parent_node_id] = eval_str

                        new_processing.append(parent_node_id)
                    
                    elif parent_node["type"]=="function":
                        func_name = parent_node["func_name"]

                        eval_node = nodes_eval[node_id]

                        eval_str = None
                        if track_origin:
                            eval_str = f'{func_name}({eval_node})@{parent_node_id}'
                        else:
                            eval_str = f'{func_name}({eval_node})'
                        
                        nodes_eval[parent_node_id] = eval_str


            processing = new_processing
        
        return nodes_eval[root_node_id]
