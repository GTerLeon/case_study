# Based on Algorithm3 IdentifyRedundancy 
# this algorithms takes as only input the conditional empirical conditional entropy w.r.t. D.
# Returns adjacency matrix A corresponding to the graphical representation of redundant data.

import networkx as nx
import numpy as np

def calculate_conditional_entropy_matrix(G):
    n = G.number_of_nodes()
    HD = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # If there's an edge from j to i, set H(Xi|Xj) = 0
                if G.has_edge(j, i):
                    HD[i,j] = 0
                else:
                    HD[i,j] = float('inf')  # No deterministic relationship
    
    return HD

def identifyRedundancy(G):
    # Get empirical conditional entropy matrix
    HD = calculate_conditional_entropy_matrix(G)
    n = len(HD)
    
    # Initialize adjacency matrix A with deterministic relationships
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if HD[i,j] == 0:
                A[i,j] = 1
    I = list(range(n))
    
    while I:
        i = I[0]

        Si = []
        for j in range(n):
            if A[i,j] == 1 and A[j,i] == 1:
                Si.append(j)
        
        # there is at least one redundant variable with i
        if len(Si) > 1:
            for k in [x for x in Si if x != i]:
                A[k,i] = 0  # no other node from the class can be parent of i
                
                for l in [x for x in Si if x not in [k, i]]:
                    A[l,k] = 0 #nodes != i form the class Si cannot be connected
            
            for node in Si:
                if node in I:
                    I.remove(node)
        else:
            # in the case of no redundancy found, remove i from I
            I.remove(i)
    
    return A

#test out on simple graph
G = nx.DiGraph()
G.add_edges_from([(0,1), (1,0), (1,2), (2,1)])  # 0 and 1 redundant variables
A = identifyRedundancy(G)
print(A)

#load dataset iris


