import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

#return dictionary where keys are features and values are lists of features that determine the functional dependencies (key features)
def discover_functional_dependencies(cond_entropy_matrix, feature_names):
    dependencies = {}
    for i, feature in enumerate(feature_names):
        dependencies[feature] = []
        for j, other_feature in enumerate(feature_names):
            if i != j and cond_entropy_matrix[i, j] == 0:
                dependencies[feature].append(other_feature)
    return dependencies

#load and preprocess iris dataset
iris_df = pd.read_csv('data/iris.csv', header=None)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
iris_df.columns = feature_names

entropy_matrix = calculate_conditional_entropy_matrix(iris_df)
dependencies = discover_functional_dependencies(entropy_matrix, feature_names)

print("Conditional Entropy Matrix:")
print(pd.DataFrame(entropy_matrix, columns=feature_names, index=feature_names))
print("\nFunctional Dependencies:")
for key, value in dependencies.items():
    print(f"{key} <- {value}")

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