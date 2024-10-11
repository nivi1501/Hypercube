import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
# Set random seed for consistent results

random.seed(42)
np.random.seed(42)
dim  =  3

# Fitness function to evaluate how close the matrix is to the original hypercube matrix
def fitness(matrix, original_matrix):
    return np.sum(np.abs(matrix - original_matrix))  # Calculate difference between the two matrices
    
# Simulated Annealing Algorithm to restore the original matrix
def simulated_annealing(matrix, original_matrix, max_iterations=500000, start_temp=10, cooling_rate=0.99):
    current_matrix = matrix.copy()
    current_fitness = fitness(current_matrix, original_matrix)
    temperature = start_temp
    
    for iteration in range(max_iterations):
        if np.array_equal(current_matrix, original_matrix): 
        #if current_fitness == 0:  # Found the original matrix
            return current_matrix, iteration
        
        # Make a small mutation (flip a random bit while keeping symmetry)
        new_matrix = current_matrix.copy()
        i, j = random.randint(0, len(matrix)-1), random.randint(0, len(matrix)-1)
        if i != j:
          new_matrix[i, j] = 1 - new_matrix[i, j]  # Flip the bit
          new_matrix[j, i] = new_matrix[i, j]  # Ensure symmetry
        
        # Calculate fitness for the new solution
        new_fitness = fitness(new_matrix, original_matrix)
        
        # Decide whether to accept the new solution
        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temperature):
            current_matrix = new_matrix
            current_fitness = new_fitness
        
        # Cool down the temperature
        temperature *= cooling_rate

    return current_matrix, max_iterations

# Provide an approximate solution (you can also input your own close-to-solution matrix here)
original_matrix = provide_approximate_solution(dim)

mergeSets=[[0,1]]

updatedMatrix, merged_matrix = mergeNodes(original_matrix, mergeSets)

# Step 3: Run simulated annealing to restore the original matrix
final_matrix, iterations = simulated_annealing(merged_matrix, original_matrix)


# Output the final matrix and number of iterations it took to find the solution
print("Initial approximate solution:")
print(merged_matrix)
print("\nFinal matrix found:")
print(final_matrix)
print(f"\nFound in {iterations} iterations")

# Visualize the matrix as a hypercube
plot_hypercube_from_adjacency_with_labels(updatedMatrix)
plot_hypercube_from_adjacency_with_labels(final_matrix)


'''
def combined_fitness(matrix, original_matrix, merged_nodes):
    # Element-wise difference (basic)
    base_fitness = np.sum(np.abs(matrix - original_matrix))
    
    # Degree-based penalty
    degrees_matrix = np.sum(matrix, axis=1)
    degrees_original = np.sum(original_matrix, axis=1)
    degree_penalty = np.sum(np.abs(degrees_matrix - degrees_original))
    
    # Penalty for merged nodes
    merged_penalty = 0
    for node in merged_nodes:
        merged_penalty += np.sum(np.abs(matrix[node, :] - original_matrix[node, :]))  # Penalty for merged rows
        merged_penalty += np.sum(np.abs(matrix[:, node] - original_matrix[:, node]))  # Penalty for merged columns
    
    return base_fitness + degree_penalty + merged_penalty    
# Fitness function to evaluate the quality of the solution
def fitness(matrix):
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    diag_elements = np.diag(matrix)

    # We want rows and columns to sum to 3, and diagonal elements to be 0
    row_fitness = np.sum(np.abs(row_sums - dim))
    col_fitness = np.sum(np.abs(col_sums - dim))
    diag_fitness = np.sum(diag_elements)

    # Lower fitness is better (penalty-based fitness function)
    return row_fitness + col_fitness + diag_fitness

# Function to find the most unfit row/column
def find_unfit_indices(matrix):
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    # Find the rows and columns whose sums are furthest from 3
    max_row_diff = np.argmax(np.abs(row_sums - dim))
    max_col_diff = np.argmax(np.abs(col_sums - dim))
    print(max_row_diff," ",max_col_diff)
    return max_row_diff, max_col_diff

# Simulated Annealing Algorithm with Intelligent Bit-Flipping
def simulated_annealing(matrix, max_iterations=2000, start_temp=10, cooling_rate=0.999):
    current_matrix = matrix.copy()
    current_fitness = fitness(current_matrix)
    temperature = start_temp
    
    for iteration in range(max_iterations):
        if current_fitness == 0:  # Found a valid solution
            return current_matrix, iteration
        
        # Find the most unfit row and column
        i, j = find_unfit_indices(current_matrix)

        # Flip the bit in the most unfit row and column (maintaining symmetry)
        #if i != j:
        new_matrix = current_matrix.copy()
        new_matrix[i, j] = 1 - new_matrix[i, j]  # Flip the bit
        new_matrix[j, i] = new_matrix[i, j]  # Ensure symmetry
        print(new_matrix)
        # Calculate fitness for the new solution
        new_fitness = fitness(new_matrix)
        
        # Decide whether to accept the new solution
        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temperature):
                current_matrix = new_matrix
                current_fitness = new_fitness
        
        # Cool down the temperature
        temperature *= cooling_rate
        
        # Early stopping: if no change in matrix over a few iterations
        if iteration > 100 and np.array_equal(current_matrix, matrix):
            print(f"Early stopping at iteration {iteration} due to no significant change.")
            break

    return current_matrix, max_iterations
'''
