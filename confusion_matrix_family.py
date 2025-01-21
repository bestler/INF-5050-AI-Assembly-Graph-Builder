import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List, Dict
from graph import Graph

# Load training data
with open('data/graphs.dat', 'rb') as file:
    graphs: List[Graph] = pickle.load(file)

all_families_set = set()

# Initialize connection matrix 
matrix_size = 95
connection_matrix = np.zeros((matrix_size, matrix_size))

# Count connections
for graph in graphs:
    edges = graph.get_edges()
    for source_node in edges:
        source_family_id = int(source_node.get_part().get_family_id())
        all_families_set.add(source_family_id)
        
        for target_node in edges[source_node]:
            target_family_id = int(target_node.get_part().get_family_id())
            all_families_set.add(target_family_id)
            
            if 1 <= source_family_id <= matrix_size and 1 <= target_family_id <= matrix_size:
                connection_matrix[source_family_id-1][target_family_id-1] += 1

# Print statistics
print(f"Total number of family connections: {int(np.sum(connection_matrix))}")
print(f"Maximum family connection frequency: {int(np.max(connection_matrix))}")
print(f"Number of unique family connections: {np.count_nonzero(connection_matrix)}")
print(f"Number of unique families: {len(all_families_set)}")

# Sort the family_ids and print them all
sorted_families = sorted(list(all_families_set))
print(f"Unique families: {sorted_families}")



# Plot heatmap
plt.figure(figsize=(20, 20)) 
sns.heatmap(connection_matrix,
            cmap='YlOrRd',
            norm=LogNorm(vmin=0.1, vmax=connection_matrix.max()),
            cbar_kws={'label': 'Connection Frequency (log scale)'},
            square=True)

plt.title('Family Connection Frequency', pad=20)
plt.xlabel('Target Family ID', labelpad=10)
plt.ylabel('Source Family ID', labelpad=10)

# Adjust tick visibility
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# Adjust layout
plt.tight_layout()

# Save and show the plot
plt.savefig('family_connections_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
