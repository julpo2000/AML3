import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

class Neat():

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.create_network(input_size, output_size)

    def create_network(self, input_size, output_size):
        network = nx.DiGraph()
        self.n_hidden_layers = 2
        self.level_sizes = [input_size, int(input_size-((input_size-output_size)/3)),  int(input_size-2*((input_size-output_size)/3)), output_size]
        node_counter = 0
        # create all nodes
        for level, level_size in enumerate(self.level_sizes):
            for i in range(level_size):
                network.add_node(node_counter, val=None, level=level)
                node_counter += 1
        
        # create all edges
        prev_level = 0
        edges = []
        for cur_level in range(1, len(self.level_sizes)):

            prev_level_sizes = 0
            for i in range(prev_level):
                prev_level_sizes += self.level_sizes[i]
            
            cur_level_sizes = prev_level_sizes + self.level_sizes[prev_level]
            for i in range(self.level_sizes[prev_level]):
                for j in range(self.level_sizes[cur_level]):
                    edges.append((i+prev_level_sizes,j+cur_level_sizes))
            prev_level = cur_level
        network.add_edges_from(edges)
        self.network = network
        #print(network.nodes(data="act"))

    def plot_network(self):
        #pos = graphviz_layout(self.network, prog='dot', args="-Grankdir=LR")
        pos = {}
        for node in self.network.nodes:
            pos[node] = [0,0]
        n_levels = len(self.level_sizes)
        prev_level_sizes = 0
        for cur_level, level_size in enumerate(self.level_sizes):
            for i in range(level_size):
                pos[i + prev_level_sizes] = [cur_level/n_levels, i/level_size]
            prev_level_sizes += level_size
        nx.draw(self.network, with_labels=True, pos=pos, font_weight='bold')
        import matplotlib.pyplot as plt
        plt.savefig("plots/plot_sine.pdf")

    def test_network(self, x_inputs, y_outputs, weight):
        network = self.network
       
        
        # give all input nodes a value
        for x_input in x_inputs:
            # load all inputs
            for i in range(self.input_size):
                network.nodes.data()[i]["val"] = x_input
            # calculate output
            for node in network.nodes.data():
                # start from last level
                if(node[1]["level"] == self.n_hidden_layers+1):
                    print(node)
                    self.recursive_network_calculate(self.n_hidden_layers+1, network, weight)
        
        print(network.nodes.data())
    
    def recursive_network_calculate(self, node_id, network, weight): 
        print(network.nodes.data(), node_id)
        # Check if parents have values
        # If they don't, go to them       
        for con_node_id in network.in_edges(node_id):
            if(network.nodes.data()[con_node_id[0]]["val"] == None):
                self.recursive_network_calculate(con_node_id[0], network, weight)

        # parents have values
        # give current node value
        if(network.nodes.data()[node_id]["val"] == None):
            print(network.nodes.data()[node_id]["val"])
            values = []
            cur_value = 0
            for con_node_id in network.in_edges(node_id):
                value = network.nodes.data()[con_node_id[0]]["val"]
                cur_value += value*weight
            network.nodes.data()[node_id]["val"] = cur_value
        

            
        


        