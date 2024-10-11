'''
This is the code for simulated annealing with
non-binary fitness function
Degree constraint is enforced
'''


import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
# Set random seed for consistent results
random.seed(42)
np.random.seed(42)

dim = 4

# Difference-based fitness function: calculate the sum of absolute differences
def difference_fitness(matrix, original_matrix):
    """Fitness function: returns the sum of absolute differences between matrices."""
    return np.sum(np.abs(matrix - original_matrix))


# Enforce row/column sum to be exactly `dim` by ensuring any flip maintains this condition
def enforce_sum_constraint(matrix):
    for i in range(len(matrix)):
        while np.sum(matrix[i, :]) != dim:
            if np.sum(matrix[i, :]) > dim:
                # If sum is greater than dim, turn random 1 into 0
                one_indices = np.where(matrix[i, :] == 1)[0]
                if len(one_indices) > 0:
                    rand_idx = random.choice(one_indices)
                    matrix[i, rand_idx] = 0
                    matrix[rand_idx, i] = 0  # Ensure symmetry
            elif np.sum(matrix[i, :]) < dim:
                # If sum is less than dim, turn random 0 into 1
                zero_indices = np.where(matrix[i, :] == 0)[0]
                zero_indices = zero_indices[zero_indices != i]  # Exclude diagonal element
                if len(zero_indices) > 0:
                    rand_idx = random.choice(zero_indices)
                    matrix[i, rand_idx] = 1
                    matrix[rand_idx, i] = 1  # Ensure symmetry
    return matrix


# Simulated Annealing Algorithm with difference-based fitness function and row/column sum constraint
def simulated_annealing_diff(matrix, original_matrix, max_iterations=1000000, start_temp=10, cooling_rate=0.9):
    current_matrix = matrix.copy()
    
    current_fitness = difference_fitness(current_matrix, original_matrix)
    temperature = start_temp
    
    for iteration in range(max_iterations):
        if np.array_equal(current_matrix, original_matrix):   # Found the original matrix
            return current_matrix, iteration, 0
        
        # Try flipping a random bit (while keeping symmetry and enforcing sum constraints)
        new_matrix = current_matrix.copy()
        i, j = random.randint(0, len(matrix) - 1), random.randint(0, len(matrix) - 1)
        
        if i != j:
            # Flip the bit
            new_matrix[i, j] = 1 - new_matrix[i, j]
            new_matrix[j, i] = new_matrix[i, j]  # Ensure symmetry
            
            # Enforce sum constraint
            new_matrix = enforce_sum_constraint(new_matrix)
        
        # Calculate fitness for the new solution
        new_fitness = difference_fitness(new_matrix, original_matrix)
        
        # Simulated annealing acceptance criteria
        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temperature):
            current_matrix = new_matrix
            current_fitness = new_fitness
        
        # Cool down the temperature
        temperature *= cooling_rate

    # If we reach here, the solution wasn't found. Compute percentage error.
    total_elements = matrix.size
    incorrect_elements = np.sum(np.abs(current_matrix - original_matrix))
    percentage_error = (incorrect_elements / total_elements) * 100
    #print("This si running")
    return current_matrix, max_iterations, percentage_error


# Define multiple merge sets for different configurations
mergeSets_list = [
    [[0, 1]],
    [[0, 1,2]],
    [[0,1,2,3]],
    [[0, 1,2, 3,4]],
    [[0, 1, 2, 3, 4, 5]],
    [[0, 1, 2, 3, 4, 5,6]],
    [[0, 1, 2, 3, 4, 5,6,7]]

]

original_matrix = provide_approximate_solution(dim)

# Automate running the code for different merge sets
for mergeSets in mergeSets_list:
    print(f"\nRunning Simulated Annealing for mergeSets: {mergeSets}")
    updatedMatrix, merged_matrix = mergeNodes(original_matrix, mergeSets)

    # Run Simulated Annealing to restore the original matrix using binary fitness function
    final_matrix, iterations, percentage_error = simulated_annealing_diff(merged_matrix, original_matrix)

    # Output the final matrix and number of iterations it took to find the solution
    print("Initial approximate solution (merged matrix):")
    print(merged_matrix)
    print("\nFinal matrix found by Simulated Annealing:")
    print(final_matrix)
    print(f"\nFound in {iterations} iterations")
    print(f"Percentage error: {percentage_error:.2f}%")
    # Visualize the matrix as a hypercube
    #plot_hypercube_from_adjacency_with_labels(original_matrix)
    #plot_hypercube_from_adjacency_with_labels(updatedMatrix)







'''
# Main Code
original_matrix = provide_approximate_solution(dim)
print("The golden solution is ")
print(original_matrix)
mergeSets = [[0, 1]]
updatedMatrix, merged_matrix = mergeNodes(original_matrix, mergeSets)

# Run Simulated Annealing to restore the original matrix using binary fitness function
final_matrix, iterations = simulated_annealing(merged_matrix, original_matrix)

# Output the final matrix and number of iterations it took to find the solution
print("Initial approximate solution (merged matrix):")
print(merged_matrix)
print("\nFinal matrix found by Simulated Annealing:")
print(final_matrix)
print(f"\nFound in {iterations} iterations")

# Visualize the matrix as a hypercube
plot_hypercube_from_adjacency_with_labels(original_matrix)
plot_hypercube_from_adjacency_with_labels(updatedMatrix)
'''
