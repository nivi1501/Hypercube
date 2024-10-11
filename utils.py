import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# Function to plot the hypercube graph using the adjacency matrix and generate binary labels
def plot_hypercube_from_adjacency_with_labels(adj_matrix):
    """Draws a hypercube using the adjacency matrix and assigns binary labels."""
    G = nx.Graph()
    n = len(adj_matrix)  # Number of nodes
    
    # Add nodes to the graph with binary labels
    labels = {i: format(i, f'0{int(np.log2(n))}b') for i in range(n)}
    
    for i in range(n):
        G.add_node(i, label=labels[i])
    
    # Add edges between nodes based on adjacency matrix
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    
    # Plot the graph using a circular layout
    pos = nx.spring_layout(G, iterations=50, seed=42)
    plt.figure(figsize=(8, 8))
    # Draw the nodes and edges first
    nx.draw(G, pos, node_color='black', node_size=500, edge_color='black')

# Draw the labels separately with white text color
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_size=12)
    plt.title(f"Hypercube Graph from Given Adjacency Matrix")
    plt.show()

# Function to merge nodes in the adjacency matrix and maintain the original dimensions
def mergeNodes(adjMatrix, mergeSets):
    """Merges specified nodes in the adjacency matrix, shifts zeroed rows and columns to the end, and returns both matrices."""
    updatedMatrix = adjMatrix.copy()  # Start with the original adjacency matrix
    
    for nodesToMerge in mergeSets:
        numNodesToMerge = len(nodesToMerge)

        if numNodesToMerge < 2:
            raise ValueError('At least two nodes must be specified for merging.')
        
        # Step 1: Merge all rows into one by OR operation for the first node
        mergedRow = np.any(updatedMatrix[nodesToMerge, :], axis=0).astype(int)
        
        # Step 2: Update the row of the first node in the merge
        updatedMatrix[nodesToMerge[0], :] = mergedRow

        # Step 3: Merge all columns into one by OR operation
        mergedColumn = np.any(adjMatrix[:, nodesToMerge], axis=1).astype(int)
        
        # Step 4: Update the column of the first node in the merge
        updatedMatrix[:, nodesToMerge[0]] = mergedColumn
        
        # Step 5: Zero out the columns and rows of the remaining nodes in the merge
        for node in nodesToMerge[1:]:
            updatedMatrix[:, node] = 0
            updatedMatrix[node, :] = 0

        # Step 6: Zero out the diagonal elements
        for i in range(updatedMatrix.shape[0]):  # Iterate over the diagonal elements
            updatedMatrix[i, i] = 0  # Set the diagonal element to zero

    # Step 6a: Ensure that all the nodes which are removed should be zero:
            # Step 5: Zero out the columns and rows of the remaining nodes in the merge
    for nodesToMerge in mergeSets:
       for node in nodesToMerge[1:]:
            updatedMatrix[:, node] = 0
            updatedMatrix[node, :] = 0 



    # Step 7: Shift zero rows and columns to the end
    zero_rows = np.all(updatedMatrix == 0, axis=1)
    non_zero_indices = np.where(~zero_rows)[0]
    zero_indices = np.where(zero_rows)[0]
    
    mergedMatrix = np.zeros_like(updatedMatrix)
    mergedMatrix[:len(non_zero_indices), :len(non_zero_indices)] = updatedMatrix[np.ix_(non_zero_indices, non_zero_indices)]
    mergedMatrix[len(non_zero_indices):, :len(non_zero_indices)] = updatedMatrix[np.ix_(zero_indices, non_zero_indices)]
    mergedMatrix[:len(non_zero_indices), len(non_zero_indices):] = updatedMatrix[np.ix_(non_zero_indices, zero_indices)]

    return updatedMatrix, mergedMatrix
    
# Function to provide an initial approximation of the solution
def provide_approximate_solution(dim):
 
    """Returns the adjacency matrix for a d-dimensional hypercube."""
    n = 2**dim
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(dim):
            matrix[i, i ^ (1 << j)] = 1
    print("The best solution is: ")
    print(matrix)        
    return matrix    
