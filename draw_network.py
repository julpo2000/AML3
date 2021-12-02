import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def importNet(fileName):
    ind = np.loadtxt(fileName, delimiter=',')
    wMat = ind[:,:-1]     # Weight Matrix
    aVec = ind[:,-1]      # Activation functions
    # Create weight key
    print(wMat, aVec)
    wVec = wMat.flatten()
    wVec[np.isnan(wVec)]=0
    wKey = np.where(wVec!=0)[0] 


    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0
    print("connection", cMat)
    return wVec, aVec, wKey

def connection_matrix(fileName):
    ind = np.loadtxt(fileName, delimiter=',')
    wMat = ind[:,:-1]     # Weight Matrix
    print(getLayer(wMat))
    wMat[wMat!=0] = 1.0
    return wMat

def plot_matrix(cMat): 
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
    print(outputs)
    
    nx.draw(network, with_labels=True, font_weight='bold')
    import matplotlib.pyplot as plt
    plt.savefig("plots/plot_sine.pdf")

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

def getNodeCoord(G,layer, nOut):
    # get input size
    temp = layer[0]
    i = 0
    while(layer[i] == temp):
        i += 1
    nIn = i

    # Calculate positions of input and output
    nNode= len(G.nodes)
    fixed_pos = np.empty((nNode,2))
    fixed_nodes = np.r_[np.arange(0,nIn),np.arange(nNode-nOut,nNode)]

    # Set Figure dimensions
    fig_wide = 10
    fig_long = 5

    # Assign x and y coordinates per layer
    x = np.ones((1,nNode))*layer # Assign x coord by layer
    x = (x/np.max(x))*fig_wide # Normalize

    _, nPerLayer = np.unique(layer, return_counts=True)

    y = cLinspace(-2,fig_long+2,nPerLayer[0])
    for i in range(1,len(nPerLayer)):
      if i%2 == 0:
        y = np.r_[y,cLinspace(0,fig_long,nPerLayer[i])]
      else:
        y = np.r_[y,cLinspace(-1,fig_long+1,nPerLayer[i])]

    fixed_pos = np.c_[x.T,y.T]
    pos = dict(enumerate(fixed_pos.tolist()))
    
    return pos

def cLinspace(start,end,N):
  if N == 1:
    return np.mean([start,end])
  else:
    return np.linspace(start,end,N)

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

def plot_matrix2(filename):
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

#cMat = connection_matrix("weightagnostic_original/prettyNeatWann/champions/biped.out")
#plot_matrix(cMat)
plot_matrix2("weightagnostic_original/prettyNeatWann/champions/biped.out")