import pickle
from collections import defaultdict
from typing import Set, Tuple, Dict

from graph import Graph
from part import Part

class SimpleEdgeFrequencyModel:
    def __init__(self):
        self.edge_frequency: Dict[Tuple[int, int], int] = defaultdict(int)

    def train(self, graphs: Set[Graph]):
        for graph in graphs:
            for node1 in graph.get_nodes():
                for node2 in graph.get_edges().get(node1, []):
                    part1_id = node1.get_part().get_part_id()
                    part2_id = node2.get_part().get_part_id()
                    self.edge_frequency[(part1_id, part2_id)] += 1
                    self.edge_frequency[(part2_id, part1_id)] += 1

    def predict_graph(self, parts: Set[Part]) -> Graph:
        graph = Graph()
        parts_list = list(parts)
        added_parts = set()

        # Create pairs of all parts and sort by frequency
        part_pairs = [(part1, part2) for i, part1 in enumerate(parts_list) for part2 in parts_list[i+1:]]
        part_pairs.sort(key=lambda pair: self.edge_frequency[(pair[0].get_part_id(), pair[1].get_part_id())], reverse=True)

        # Add edges based on frequency
        for part1, part2 in part_pairs:
            if part1 not in added_parts or part2 not in added_parts:
                graph.add_undirected_edge(part1, part2)
                added_parts.update([part1, part2])

        # Ensure all parts are integrated
        for part in parts:
            if part not in added_parts:
                # Create an arbitrary edge with an already added part
                graph.add_undirected_edge(part, next(iter(added_parts)))
                added_parts.add(part)

        return graph

def load_model(file_path: str) -> SimpleEdgeFrequencyModel:
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


if __name__ == '__main__':
    # Example usage
    with open('data/train_graphs.dat', 'rb') as file:
        train_graphs: Set[Graph] = pickle.load(file)


    model = SimpleEdgeFrequencyModel()
    model.train(train_graphs)

    # Save the model
    with open('simple_edge_frequency_model.pkl', 'wb') as file:
        pickle.dump(model, file)