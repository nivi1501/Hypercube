import random
import itertools

# Constants
TOTAL_VIRTUAL_NODES = 256
BYTE_RANGE = 256
VIRTUAL_NODES_PER_PHYSICAL = 16 # Adjust this to 2, 4, 8, or 16 as needed
TOTAL_PHYSICAL_NODES = int(TOTAL_VIRTUAL_NODES/VIRTUAL_NODES_PER_PHYSICAL)
TOTAL_PHYSICAL_NODES_ACTUAL = 262144
PHYSICAL_NODES_PER_VIRTUAL = 16  # Adjust this to 0, 2, 4, 8, or 16 as needed (PN per VN)

iterations = 25000

# Fix the seed for reproducibility
random.seed(40)

def find_common_values_among_all_with_convergence(keys):
    # Start with the first list's values as the initial common values
    common_values = set(keys[0])
    convergence_steps = [len(common_values)]  # Track the size of common values initially

    i=1
    # Iteratively intersect each list and track the convergence
    for key_list in keys[1:]:
        common_values.intersection_update(key_list)  # Keep only values common to all sets
        convergence_steps.append(len(common_values))  # Track the size after each intersection
        if(len(common_values) == 1):
           print("Convergence acheived at iteration , ", i)
           return list(common_values), convergence_steps       
        i=i+1
    return list(common_values), convergence_steps  # Return final common values and convergence steps


# Function to generate the actual random virtual to physical mapping of the T-table
def generateActualMapping():
    # Creating a list of virtual nodes and shuffling it to randomize their order
    virtual_nodes = list(range(TOTAL_VIRTUAL_NODES))
    random.shuffle(virtual_nodes)

    # Initialize the mapping dictionary
    virtual_to_physical_map = {}

    # Map the randomized virtual nodes to physical nodes
    # Out of 256 physical nodes, only some nodes have actual VN values
    for i in range(TOTAL_PHYSICAL_NODES):
        # Each physical node will be assigned a specific number of virtual nodes from the shuffled list
        start_index = i * VIRTUAL_NODES_PER_PHYSICAL
        virtual_to_physical_map[i] = virtual_nodes[start_index:start_index + VIRTUAL_NODES_PER_PHYSICAL]

    # Add noise to the process
    # Iterate across all the VNs
    for virtual_node in range(TOTAL_VIRTUAL_NODES):
        # Randomly select some physical nodes to add noise
        # Noise can be added to any of the physical nodes
        selected_physical_nodes = random.sample(range(TOTAL_PHYSICAL_NODES_ACTUAL), PHYSICAL_NODES_PER_VIRTUAL)
        # Add the same VN to all the selected physical nodes
        for physical_node in selected_physical_nodes:
            if physical_node in virtual_to_physical_map:
                virtual_to_physical_map[physical_node].append(virtual_node)
            else:
                virtual_to_physical_map[physical_node] = [virtual_node]
        
    # Uncomment to print the mapping for verification
    #for physical_node, virtual_nodes in virtual_to_physical_map.items():
    #     print(f"Physical Node {physical_node}: Virtual Nodes {virtual_nodes}")
    
    return virtual_to_physical_map



# Function to map the 256 plaintext bytes to the given physical nodes by the attacker using the unknown second key byte        
def processingKeyBytes(key, actual_virtual_to_physical_map):
    # Initialize a dictionary to store the plaintext-physical node mapping
    

    SelectedKeyBytes = [key[0], key[1]]
    physical_node_to_plaintext_map_keys = []
    for keyByte in SelectedKeyBytes:
     physical_node_to_plaintext_map = {}
    # Vary the first plaintext byte from 0 to 255
     for plaintext_byte in range(BYTE_RANGE):
        # Calculate the virtual node as plaintext XOR key
        virtual_node = plaintext_byte ^ keyByte

        # Search for the physical node in the actual mapping using the hidden VN
        physical_node = None
        for node, virtuals in actual_virtual_to_physical_map.items():
            if virtual_node in virtuals:
                physical_node = node
                #break   # Earlier it was one PN for a VN, but now the attacker needs to check mltiple PNs

                # Record the plaintext associated with each physical node
                if physical_node is not None:
                    # If the mapping does not exist, create a new entry
                    if physical_node not in physical_node_to_plaintext_map:
                        physical_node_to_plaintext_map[physical_node] = [plaintext_byte]
                    else:
                        # If the mapping exists, append the plaintext byte
                        physical_node_to_plaintext_map[physical_node].append(plaintext_byte)
     physical_node_to_plaintext_map_keys.append(physical_node_to_plaintext_map)   
    return physical_node_to_plaintext_map_keys




def find_common_plaintext_lists(physical_node_to_plaintext_map_keys):
    # Extract the two mappings from the provided list
    map1 = physical_node_to_plaintext_map_keys[0]
    map2 = physical_node_to_plaintext_map_keys[1]
    
    # Dictionary to store the common plaintext lists for matching physical nodes
    common_plaintext_lists = {}

    # Iterate over each physical node in map1
    for pn in map1:
        # Check if the physical node exists in both maps
        if pn in map2:
            # If they exist in both, store the lists from both maps for that physical node
            common_plaintext_lists[pn] = {
                "map1_plaintexts": map1[pn],
                "map2_plaintexts": map2[pn]
            }

    return common_plaintext_lists



def find_keys(common_plaintext_lists,known_key):
    keys = []  # Dictionary to store possible K1 and K2 values for each physical node
    k1_xor_candidates = set()
    for pn, plaintexts in common_plaintext_lists.items():
        map1_plaintexts = plaintexts['map1_plaintexts']
        map2_plaintexts = plaintexts['map2_plaintexts']
        # Iterate over all pairs in map1_plaintexts and map2_plaintexts to find key XOR values
        #P1^K1=P2^K2 ---> P1^P2 = K1^K2
        for P1 in map1_plaintexts:
            for P2 in map2_plaintexts:
                if P1 != P2:
                    candidate_k1 = P1 ^ P2
                    k1_xor_candidates.add(candidate_k1)

    # Multiple possible candidates, even if I assume that one keybyte is known
    for xor_result in k1_xor_candidates:
        keys.append(known_key^xor_result)
        
    return keys  # Returning keys dictionary as it might be used later in the code



# Main code starts here
key = bytes(random.getrandbits(2) for _ in range(16))
targetKeyByte = 1
print("The actual key byte value at position 0 is ",key[0] )
print("The actual key byte value at position 1 is ",key[targetKeyByte] )


keys = []
for i in range(iterations):
        # Generate the actual mapping
        actual_virtual_to_physical_map = generateActualMapping()

        # Process the plaintext to physical node mapping for the second key byte
        physical_node_to_plaintext_map = processingKeyBytes(key, actual_virtual_to_physical_map)

        common_plaintext_list = find_common_plaintext_lists(physical_node_to_plaintext_map)

        keys.append(find_keys(common_plaintext_list,key[0]))

# Assuming `keys` is a list of lists, where each inner list contains key candidates
common_values, convergence_steps = find_common_values_among_all_with_convergence(keys)
#print("Values common to all arrays:", common_values)
#print("Convergence steps (size reduction at each step):", convergence_steps)


