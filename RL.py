import torch
import numpy as np
import random
from utils import *

# Hyperparameters for Q-learning
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.3     # Epsilon-greedy exploration rate
episodes = 100   # Number of episodes for training

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU.")


# Function to enforce the degree constraint on GPU using PyTorch tensors
def enforce_sum_constraint(matrix, dim):
    for i in range(matrix.shape[0]):
        while torch.sum(matrix[i, :]) != dim:
            if torch.sum(matrix[i, :]) > dim:
                # If sum is greater than dim, turn random 1 into 0
                one_indices = torch.where(matrix[i, :] == 1)[0]
                if one_indices.size(0) > 0:
                    rand_idx = random.choice(one_indices.cpu()).item()
                    matrix[i, rand_idx] = 0
                    matrix[rand_idx, i] = 0  # Ensure symmetry
            elif torch.sum(matrix[i, :]) < dim:
                # If sum is less than dim, turn random 0 into 1
                zero_indices = torch.where(matrix[i, :] == 0)[0]
                zero_indices = zero_indices[zero_indices != i]  # Exclude diagonal element
                if zero_indices.size(0) > 0:
                    rand_idx = random.choice(zero_indices.cpu()).item()
                    matrix[i, rand_idx] = 1
                    matrix[rand_idx, i] = 1  # Ensure symmetry
    return matrix

# Define the reward function based on the fitness difference (on GPU)
def reward_function(current_matrix, original_matrix):
    return -torch.sum(torch.abs(current_matrix - original_matrix))


# Initialize the Q-table (state-action values) on GPU
def initialize_q_table(n):
    return torch.zeros((n, n, 2), device=device)  # Two possible actions for each state (flip or no flip)

# Q-learning algorithm for reconstructing the adjacency matrix (GPU)
def q_learning(matrix, original_matrix, dim, max_iterations=10000):
    q_table = initialize_q_table(matrix.shape[0])  # Initialize Q-table
    current_matrix = matrix.clone()
    total_steps = 0  # Track total iterations    
    for episode in range(episodes):
        # Epsilon-greedy action selection
        for iteration in range(max_iterations):
            i, j = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[0] - 1)
            if i != j:
                if random.uniform(0, 1) < epsilon:
                    # Explore: Flip the bit randomly
                    action = random.choice([0, 1])
                else:
                    # Exploit: Select the best action from Q-table
                    action = torch.argmax(q_table[i, j]).item()

                # Apply the action
                if action == 1:
                    current_matrix[i, j] = 1 - current_matrix[i, j]
                    current_matrix[j, i] = current_matrix[i, j]  # Ensure symmetry

                    # Enforce the degree constraint
                    current_matrix = enforce_sum_constraint(current_matrix, dim)

                # Get reward and update Q-table
                reward = reward_function(current_matrix, original_matrix)
                best_future_q = torch.max(q_table[i, j])
                q_table[i, j, action] = q_table[i, j, action] + alpha * (reward + gamma * best_future_q - q_table[i, j, action])

                # Check if the matrix is correct
                if torch.equal(current_matrix, original_matrix):
                    total_steps = (episode * max_iterations) + iteration + 1
                    return current_matrix.cpu(), total_steps, 0
        total_steps += max_iterations
    # Calculate the percentage error if the solution is not found
    total_elements = matrix.numel()
    incorrect_elements = torch.sum(torch.abs(current_matrix - original_matrix)).item()
    percentage_error = (incorrect_elements / total_elements) * 100

    return current_matrix.cpu(), total_steps, percentage_error

















# Example usage
# Define multiple merge sets for different configurations
mergeSets_list = [
    [[0, 1]],
    [[0, 1, 2]],
    [[0, 1 ,2, 3]],
    [[0, 1, 2, 3, 4]],
    [[0, 1, 2, 3, 4, 5]],
    [[0, 1, 2, 3, 4, 5, 6]],
    [[0, 1, 2, 3, 4, 5, 6, 7]]
]

dim = 4
original_matrix = torch.tensor(provide_approximate_solution(dim), dtype=torch.float32, device=device)

# Automate running the code for different merge sets
for mergeSets in mergeSets_list:
    print(f"\nRunning Simulated Annealing for mergeSets: {mergeSets}")
    updatedMatrix, merged_matrix_np = mergeNodes(original_matrix.cpu().numpy(), mergeSets)
    merged_matrix = torch.tensor(merged_matrix_np, dtype=torch.float32, device=device)

    # Run Q-learning to restore the original matrix on GPU
    final_matrix, iterations, percentage_error = q_learning(merged_matrix, original_matrix, dim)
    #final_matrix, iterations, percentage_error = q_learning_binary(merged_matrix, original_matrix, dim)


    # Output the final matrix and number of iterations it took to find the solution
    #print("Initial approximate solution (merged matrix):")
    #print(merged_matrix)
    #print("\nFinal matrix found by Simulated Annealing:")
    #print(final_matrix)
    print(f"\nFound in {iterations} iterations")
    print(f"Percentage error: {percentage_error:.2f}%")
    # Visualize the matrix as a hypercube
    #plot_hypercube_from_adjacency_with_labels(original_matrix)
    #plot_hypercube_from_adjacency_with_labels(updatedMatrix)


'''

original_matrix = torch.tensor(provide_approximate_solution(dim), dtype=torch.float32, device=device)
mergeSets = [[0, 1], [1, 2, 3]]  # Example merging sets

# Convert original_matrix and merged_matrix to torch tensors
updatedMatrix, merged_matrix_np = mergeNodes(original_matrix.cpu().numpy(), mergeSets)
merged_matrix = torch.tensor(merged_matrix_np, dtype=torch.float32, device=device)

# Run Q-learning to restore the original matrix on GPU
final_matrix, iterations, percentage_error = q_learning(merged_matrix, original_matrix, dim)

# Output the results
print("Final matrix after Q-learning:")
print(final_matrix)
print(f"Iterations: {iterations}")
print(f"Percentage error: {percentage_error:.2f}%")
'''
