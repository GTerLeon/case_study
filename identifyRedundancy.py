# Based on Algorithm3 IdentifyRedundancy 
# this algorithms takes as only input the conditional empirical conditional entropy w.r.t. D.
# Returns adjacency matrix A corresponding to the graphical representation of redundant data.

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from scipy.stats import entropy 

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
            # (idk should add this be here to avoid self-loops)  i != j:
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
                print(f"Removing edge ({k} -> {i})")
                A[k,i] = 0  # no other node from the class can be parent of i
                
                for l in [x for x in Si if x not in [k, i]]:
                    print(f"Removing edge ({l} -> {k})")
                    A[l,k] = 0 #nodes != i form the class Si cannot be connected
            
            for node in Si:
                if node in I:
                    I.remove(node)
        else:
            # in the case of no redundancy found, remove i from I
            I.remove(i)
    
    return A

# # #test out on simple graph
# G = nx.DiGraph()
# G.add_edges_from([(0,1), (1,0), (1,2), (2,1), (2,3), (3,2), (3,4)])  # 0 and 1 redundant variables
# A = identifyRedundancy(G)
# print(A)


# Load and preprocess iris dataset
iris_data = pd.read_csv("data/iris.csv")
iris_data.columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'class']
label_encoder = LabelEncoder()
iris_data['class'] = label_encoder.fit_transform(iris_data['class'])

scaler = MinMaxScaler()
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
iris_data[features] = scaler.fit_transform(iris_data[features])

G = nx.DiGraph()
for i in range(len(features)):
    G.add_node(i)

A = identifyRedundancy(G)

print(f"Adjacency matrix: \n", A)

# Code MAI
# #Create the graph and calculate conditional entropy matrix
# def calculate_pairwise_entropy(data):
#     n_features = data.shape[1]
#     entropy_matrix = np.zeros((n_features, n_features))
    
#     for i in range(n_features):
#         for j in range(n_features):
#             if i != j:
#                 # Conditional entropy calculation placeholder
#                 # Replace with actual calculation of H(X_i | X_j)
#                 entropy_matrix[i, j] = np.random.rand() 
#             else:
#                 entropy_matrix[i, j] = 0  # Self-entropy is 0
#     return entropy_matrix

# # Step 3: Create the graph based on the entropy matrix
# def create_graph_from_entropy(matrix, threshold=0.2):
#     n = matrix.shape[0]
#     G = nx.DiGraph()
#     for i in range(n):
#         for j in range(n):
#             if matrix[i, j] < threshold:  # Threshold for deterministic relationships
#                 G.add_edge(j, i)  # Add edge based on entropy condition
#     return G


