import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def getLayer(wMat):
  '''
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1
  '''
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while(True): # Loop until sorting doesn't help any more
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
        break
  return layer-1

# get positions for each node dependend on layers
def get_pos(layer, outputs):
    values, nodes_per_layer = np.unique(layer, return_counts=True)
    
    pos = {}
    cur_layer_i = 1
    prev_layer = 0
    for i in range(len(layer)):
        cur_layer = layer[i]
        if(cur_layer != prev_layer):
            cur_layer_i = 1
            prev_layer = cur_layer
        else:
            cur_layer_i += 1
        x_pos = cur_layer_i/(nodes_per_layer[int(cur_layer)]+1)
        pos[i] = [layer[i], x_pos]
    return pos

def plot_matrix(filename):
    # get input
    ind = np.loadtxt(filename, delimiter=',')
    wMat = ind[:,:-1]     # Weight Matrix
    cMat = wMat
    cMat[cMat!=0] = 1.0

    network = nx.DiGraph()
    # add all nodes
    for i in range(len(cMat)):
        network.add_node(i)
    
    # add connections
    outputs = []
    for i in range(len(cMat)):
        is_output = True       
        for j in range(len(cMat[0])):
            if(cMat[i][j] == 1.0):
                network.add_edge(i,j)
                is_output = False
        if(is_output):
            outputs.append(i)

    # get positions of nodes from built in functions
    layer = getLayer(wMat)

    # change output layer
    prev_highest = layer[outputs[0]-1]
    for nodeID in outputs:
        layer[nodeID] = prev_highest+1
    
    pos = get_pos(layer, outputs)
        
    nx.draw(network, with_labels=True, pos=pos, font_weight='bold')
    import matplotlib.pyplot as plt
    plt.savefig("plots/network.pdf")


plot_matrix("weightagnostic_original/prettyNeatWann/champions/biped.out")