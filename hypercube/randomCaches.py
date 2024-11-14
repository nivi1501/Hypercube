import random
import itertools

# Constants
TOTAL_VIRTUAL_NODES = 256
BYTE_RANGE = 256
VIRTUAL_NODES_PER_PHYSICAL = 2 # Adjust this to 2, 4, 8, or 16 as needed
TOTAL_PHYSICAL_NODES = int(TOTAL_VIRTUAL_NODES/VIRTUAL_NODES_PER_PHYSICAL)
TOTAL_PHYSICAL_NODES_ACTUAL = 256
PHYSICAL_NODES_PER_VIRTUAL = 0  # Adjust this to 0, 2, 4, 8, or 16 as needed (PN per VN)

# Fix the seed for reproducibility
random.seed(42)

# Extra helper function to check if the attacker found the actual mapping accurately
def compare_mappings(actual_map, attacker_map):
    if set(actual_map.keys()) != set(attacker_map.keys()):
        return False
    for physical_node in actual_map:
        # Sort and deduplicate both lists for comparison
        if sorted(set(actual_map[physical_node])) != sorted(set(attacker_map.get(physical_node, []))):
            return False
    return True



# Function to generate the actual random virtual to physical mapping of the T-table
def generateActualMappingNoiseFree():
    # Creating a list of virtual nodes and shuffling it to randomize their order
    virtual_nodes = list(range(TOTAL_VIRTUAL_NODES))
    random.shuffle(virtual_nodes)

    # Initialize the mapping dictionary
    virtual_to_physical_map = {}

    # Map the randomized virtual nodes to physical nodes
    for i in range(TOTAL_PHYSICAL_NODES):
        # Each physical node will be assigned a specific number of virtual nodes from the shuffled list
        start_index = i * VIRTUAL_NODES_PER_PHYSICAL
        virtual_to_physical_map[i] = virtual_nodes[start_index:start_index + VIRTUAL_NODES_PER_PHYSICAL]

    
    # Uncomment to print the mapping for verification
    #for physical_node, virtual_nodes in virtual_to_physical_map.items():
    #     print(f"Physical Node {physical_node}: Virtual Nodes {virtual_nodes}")
    
    return virtual_to_physical_map



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
        # Noise can be added to any of the 256 physical nodes
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


# Function to generate the virtual node to physical node mapping inferred by the attacker
# using the known key byte and plaintext
def findingVirtualMappingByAttacker(keyByte, actual_virtual_to_physical_map):
    # Initialize a dictionary for the attacker's inferred virtual-to-physical mapping
    attacker_virtual_to_physical_map = {}

    # Iterate over all possible plaintext bytes (0-255)
    for plaintext_byte in range(256):
        # Calculate the virtual node as plaintext XOR key
        virtual_node = plaintext_byte ^ keyByte

        # Find physical nodes associated with this virtual node in the actual mapping
        for physical_node, virtual_nodes in actual_virtual_to_physical_map.items():
            # An attacker observed a cache hit
            if virtual_node in virtual_nodes:
                if physical_node not in attacker_virtual_to_physical_map:
                    attacker_virtual_to_physical_map[physical_node] = [virtual_node]
                else:
                    if virtual_node not in attacker_virtual_to_physical_map[physical_node]:
                        attacker_virtual_to_physical_map[physical_node].append(virtual_node)

    return attacker_virtual_to_physical_map



# Function to map the 256 plaintext bytes to the given physical nodes by the attacker using the unknown second key byte        
def processingSecondKeyByte(keyByte, actual_virtual_to_physical_map):
    # Initialize a dictionary to store the plaintext-physical node mapping
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

    return physical_node_to_plaintext_map




# Analysis of all possible pairs for each physical node
def caseAnalysis(physical_node_to_plaintext_map, attacker_virtual_to_physical_map):  

    # Initialize a dictionary to store key candidates for each physical node
    node_key_candidates = {}

    for physical_node, plaintexts in physical_node_to_plaintext_map.items():
        # Check if physical node is in the attacker's mapping
        if physical_node in attacker_virtual_to_physical_map:
            virtualNodes = attacker_virtual_to_physical_map[physical_node]
            
            print(f"Physical Node: {physical_node}")
            print(f"Plaintext values: {plaintexts}")
            #print("Possible virtual node and plaintext combinations with resulting key values:")

            # List to store unique key candidates for this physical node
            key_candidates = set()

            # For each virtual node and plaintext combination, calculate possible key values
            i=0
            for V in virtualNodes:
                for P in plaintexts:
                    #P = plaintexts[1] # Uncomment this line and comment the above line
                                       # to generate results for a single plaintext
                    # Calculate the key candidate as P ^ V
                    key_candidate = P ^ V
                    key_candidates.add(key_candidate)
                    i=i+1
                  
                    #print(f"  Virtual Node: V={V}, Plaintext: P={P} -> Key Candidate: {key_candidate}")


            # Store the key candidates for the current physical node
            node_key_candidates[physical_node] = key_candidates

            # Print all unique key candidates for the physical node
            print("Total possible key candidates for a physical node is ",i)  
            print("Unique possible key candidates for a physical node is ",len((key_candidates)))  
            print(f"Unique key candidates for Physical Node {physical_node}: {sorted(key_candidates)}")
            print()  # Blank line for readability

    return node_key_candidates



# Extra helper function to find the common key candidate across all physical nodes
def find_common_key_candidate(node_key_candidates):
    # Start with the key candidates of the first physical node
    common_candidates = None
    i=0
    for physical_node, candidates in node_key_candidates.items():
        if common_candidates is None:
            common_candidates = candidates  # Initialize with the first node's candidates
        else:
            # Intersect with candidates from the current physical node
            common_candidates &= candidates
            print("Common candidates are ",common_candidates, " for iteration i=",i)
            if(len(common_candidates)==1):
               break
        # If no common candidates remain, break early
        if not common_candidates:
            print("The attack failed, there are no common candidates ")
            break
        i=i+1
    # Return the common candidates, if any
    return common_candidates





# Main code starts here
key = bytes(random.getrandbits(8) for _ in range(16))

# Generate the actual mapping
actual_virtual_to_physical_map = generateActualMapping()

# Find the VN to Physical node mapping for the attacker
attacker_virtual_to_physical_map = findingVirtualMappingByAttacker(key[0], actual_virtual_to_physical_map)

for physical_node, virtual_nodes in attacker_virtual_to_physical_map.items():
         print(f"Physical Node {physical_node}: Virtual Nodes {virtual_nodes}")



# Check mapping accuracy
if compare_mappings(actual_virtual_to_physical_map, attacker_virtual_to_physical_map):
    print("The mappings are equal. The attack was a success.")
else:
    print("Attacker not able to find the mapping as the mappings are not equal.")


targetKeyByte = 1
print("The actual key byte value is ",key[targetKeyByte] )

# Uncomment to perform re-keying
#random.seed(100)
#actual_virtual_to_physical_map = generateActualMapping()

# Process the plaintext to physical node mapping for the second key byte
physical_node_to_plaintext_map = processingSecondKeyByte(key[targetKeyByte], actual_virtual_to_physical_map)


for physical_node, virtual_nodes in physical_node_to_plaintext_map.items():
         print(f"Physical Node {physical_node}: Plaintext {virtual_nodes}")


# Perform case analysis
node_key_candidates = caseAnalysis(physical_node_to_plaintext_map, attacker_virtual_to_physical_map)

# Find the common key candidate across all physical nodes
common_key_candidates = find_common_key_candidate(node_key_candidates)

# Print the result
if common_key_candidates:
    print(f"Common key candidate(s) across all physical nodes: {sorted(common_key_candidates)}")
else:
    print("No common key candidate found across all physical nodes.")
    
    
print(f"Common key candidate(s) across all physical nodes: {len(common_key_candidates)}")
    
print("The actual key byte value is ",key[targetKeyByte] )    























'''



# Function to generate the virtual node to physical node mapping by the attacker
# using the known key byte and plaintext
def findingVirtualMappingByAttacker(keyByte, actual_virtual_to_physical_map):
    # Initialize a dictionary to store the attacker's virtual-to-physical node mapping
    attacker_virtual_to_physical_map = {}
    
    # Vary the first plaintext byte from 0 to 255
    for plaintext_byte in range(256):  # Assuming BYTE_RANGE is 256
        # Calculate the virtual node (VN) as plaintext XOR key
        virtual_node = plaintext_byte ^ keyByte


        # Maintain the correct mapping
        # Search for the physical node in the actual mapping
        physical_node = None
        for phyNode, virtuals in actual_virtual_to_physical_map.items():
            if virtual_node in virtuals:
                physical_node = phyNode
                break

        # Create the attacker's mapping
        if physical_node is not None:
            # If the mapping does not exist, create a new entry
            if physical_node not in attacker_virtual_to_physical_map:
                attacker_virtual_to_physical_map[physical_node] = [virtual_node]
            else:
                # If the mapping exists, append the virtual node
                attacker_virtual_to_physical_map[physical_node].append(virtual_node)


        # Adding noise to the correct mapping
        # Nmber of physical nodes are also random
        if (PHYSICAL_NODES_PER_VIRTUAL !=0 ):
                num_mappings = random.randint(1, PHYSICAL_NODES_PER_VIRTUAL)
                possible_physical_nodes = list(range(TOTAL_PHYSICAL_NODES))
        
                # Randomly select a number of extra physical nodes to map this VN to        
                selected_physical_nodes = random.sample(possible_physical_nodes, min(num_mappings, len(possible_physical_nodes)))

                # Create the attacker's mapping with multiple mappings
                # For the extra physical nodes (additional cache hits)
                for physical_node in selected_physical_nodes:
                # If the physical node mapping does not exist, create a new entry
                        if physical_node not in attacker_virtual_to_physical_map:
                                attacker_virtual_to_physical_map[physical_node] = [virtual_node]
                        else:
                                # If the mapping exists, append the virtual node if not already present
                                if virtual_node not in attacker_virtual_to_physical_map[physical_node]:
                                        attacker_virtual_to_physical_map[physical_node].append(virtual_node)

    # Check and print virtual nodes mapped to multiple physical nodes
    # Dictionary to count occurrences of each virtual node
    virtual_node_count = {}
    max_physical_nodes = 0  # Variable to track the maximum number of PNs attached to any VN
    max_virtual_node = None  # Variable to track the VN with the maximum PN mappings

    # Count the number of times each virtual node is mapped
    for physical_nodes in attacker_virtual_to_physical_map.values():
        for virtual_node in physical_nodes:
            if virtual_node in virtual_node_count:
                virtual_node_count[virtual_node] += 1
            else:
                virtual_node_count[virtual_node] = 1

    # Print virtual nodes mapped to multiple physical nodes
    #print("Virtual nodes mapped to multiple physical nodes:")
    for virtual_node, count in virtual_node_count.items():
        #if count > 1:
        #    print(f"Virtual Node {virtual_node} is mapped to {count} physical nodes.")

        # Update the maximum if this virtual node has more physical node mappings
        if count > max_physical_nodes:
            max_physical_nodes = count
            max_virtual_node = virtual_node

    # Print the maximum number of physical nodes attached to a single virtual node
    if max_virtual_node is not None:
        print(f"Maximum number of physical nodes mapped to a single virtual node: "
              f"Virtual Node {max_virtual_node} ID with {max_physical_nodes} physical nodes.")



    return attacker_virtual_to_physical_map
'''
