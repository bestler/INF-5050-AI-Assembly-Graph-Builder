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

all_nodes_set = set()

# Initialize connection matrix
matrix_size = 2270
connection_matrix = np.zeros((matrix_size, matrix_size))

# Count connections
for graph in graphs:
    edges = graph.get_edges()
    for source_node in edges:
        source_part_id = int(source_node.get_part().get_part_id())
        all_nodes_set.add(source_part_id)
        
        for target_node in edges[source_node]:
            target_part_id = int(target_node.get_part().get_part_id())
            all_nodes_set.add(target_part_id)
            
            if 1 <= source_part_id <= matrix_size and 1 <= target_part_id <= matrix_size:
                connection_matrix[source_part_id-1][target_part_id-1] += 1

# Print statistics
print(f"Total number of connections: {int(np.sum(connection_matrix))}")
print(f"Maximum connection frequency: {int(np.max(connection_matrix))}")
print(f"Number of unique connections: {np.count_nonzero(connection_matrix)}")
print(f"Number of unique nodes: {len(all_nodes_set)}")

# Remove rows/columns with low activity
row_sums = connection_matrix.sum(axis=1)
col_sums = connection_matrix.sum(axis=0)

# Keep parts with more than 10 connections (adjust threshold as needed)
threshold = 50
active_rows = np.where(row_sums > threshold)[0]
active_cols = np.where(col_sums > threshold)[0]

# Reduce matrix
reduced_matrix = connection_matrix[active_rows][:, active_cols]

# Sort by total connection frequency
row_sort_idx = np.argsort(reduced_matrix.sum(axis=1))[::-1]
col_sort_idx = np.argsort(reduced_matrix.sum(axis=0))[::-1]
reduced_matrix = reduced_matrix[row_sort_idx][:, col_sort_idx]

# Update labels
active_rows = active_rows[row_sort_idx]
active_cols = active_cols[col_sort_idx]
row_labels = [f"{i+1}" for i in active_rows]
col_labels = [f"{i+1}" for i in active_cols]

#row_labels = [f"Part {i+1}" for i in active_rows]
#col_labels = [f"Part {i+1}" for i in active_cols]

# Plot heatmap
plt.figure(figsize=(100, 100))
sns.heatmap(reduced_matrix,
            cmap='YlOrRd',
            norm=LogNorm(vmin=0.1, vmax=reduced_matrix.max()),
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Connection Frequency (log scale)'},
            square=True)

plt.title('Filtered Part Connection Frequency (Most Active Connections)', pad=20)
plt.xlabel('Target Part ID', labelpad=10)
plt.ylabel('Source Part ID', labelpad=10)

# Adjust tick visibility
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# Adjust layout
plt.tight_layout()

# Save and show the plot
plt.savefig('part_connections_heatmap_improved_v2.png', dpi=300, bbox_inches='tight')
plt.show()