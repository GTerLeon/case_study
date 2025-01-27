import itertools
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from scipy.stats import entropy

def conditional_entropy(x, y):
    joint_prob = pd.crosstab(x, y, normalize=True)  #joint probability P(X, Y)
    y_marginals = joint_prob.sum(axis=0)  #marginal probabilities P(Y)
    cond_entropy = 0
    
    for y_val, p_y in y_marginals.items():
        p_x_given_y = joint_prob.loc[:, y_val] / p_y  #P(X|Y=y)
        cond_entropy += p_y * entropy(p_x_given_y, base=2)
    return cond_entropy
    
#calculate conditional entropy matrix for a given dataset
def calculate_conditional_entropy_matrix(data):
    data = data.apply(LabelEncoder().fit_transform)
    n = data.shape[1]
    HD = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                HD[i, j] = conditional_entropy(data.iloc[:, i], data.iloc[:, j])
    return HD

#Detect redundancy based on conditional entropy close to 0, assign one representative for each redundancy class
def handle_redundancies(cond_entropy_matrix, feature_names):
    n = cond_entropy_matrix.shape[0]
    is_redundant = np.isclose(cond_entropy_matrix, 0) & np.isclose(cond_entropy_matrix.T, 0)
    visited = np.zeros(n, dtype=bool)
    representatives = {}  # Stores the representative for each class

    for i in range(n):
        if not visited[i]:
            # Find all variables redundant with feature `i`
            redundancy_class = [j for j in range(n) if is_redundant[i, j]]
            representative = redundancy_class[0]  # Choose the first as representative
            representatives[representative] = redundancy_class
            # Mark all redundant variables as visited
            for idx in redundancy_class:
                visited[idx] = True

    return representatives

#return dictionary where keys are features and values are lists of features that determine the functional dependencies (key features)
def discover_functional_dependencies(cond_entropy_matrix, feature_names, threshold=0.6):
    dependencies = {}
    n = cond_entropy_matrix.shape[0]
    representatives = handle_redundancies(cond_entropy_matrix, feature_names)

    for rep, redundant_group in representatives.items():
        rep_feature = feature_names[rep]
        dependencies[rep_feature] = []
        
        # Check functional dependencies for the representative
        for j in range(n):
            if j != rep and cond_entropy_matrix[rep, j] < threshold:
                dependencies[rep_feature].append(feature_names[j])

        # Add redundant variables as dependent on their representative
        for redundant_var in redundant_group:
            if redundant_var != rep:  # Exclude the representative itself
                dependencies[feature_names[redundant_var]] = [rep_feature]

    return dependencies

# # Load the Iris dataset
# iris = load_iris()
# iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_df['target'] = iris.target  # Add target as a column

# # Calculate conditional entropy matrix and discover dependencies
# feature_names = iris_df.columns
# entropy_matrix = calculate_conditional_entropy_matrix(iris_df)
# dependencies = discover_functional_dependencies(entropy_matrix, feature_names)

# # Display results
# print("\nConditional Entropy Matrix:")
# print(pd.DataFrame(entropy_matrix, columns=feature_names, index=feature_names))
# print("\nFunctional Dependencies:")
# for key, value in dependencies.items():
#     print(f"{key} <- {value}")

# G = pd.DataFrame({
#     'A': [1, 2, 3, 1, 2, 3],
#     'B': [10, 20, 30, 10, 20, 30]
# })

# feature_names = ['A', 'B']

# entropy_matrix = calculate_conditional_entropy_matrix(G)
# dependencies = discover_functional_dependencies(entropy_matrix, feature_names)

# print("\nConditional Entropy Matrix:")
# print(pd.DataFrame(entropy_matrix, columns=feature_names, index=feature_names))
# print("\nFunctional Dependencies:")
# for key, value in dependencies.items():
#     print(f"{key} <- {value}")