from typing import Set
import pickle
from prediction_classes.my_prediction_model import MyPredictionModel
from graph import Graph
from part import Part
from node import Node
import torch
import torch.nn as nn
import networkx as nx


class PartPredictor(MyPredictionModel):
    
    def __init__(self):
        super().__init__()
        self.model = None
    
    
    def load_model(self, file_path: str) -> MyPredictionModel:
        """
        Loads the model from the given file path.
        :param file_path: path to the file
        :return: the loaded model
        """

        # Load from pytorch dict state
        model = torch.load(file_path)
        self.model = model

        return self

    
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Predicts the graph based on the given parts.
        :param parts: set of parts to form up an assembly (i.e. a graph)
        :return: graph
        """

        part_ids = [int(part.get_part_id()) for part in parts]
        probabilities = self.calculate_probabilities(self.model, part_ids)
        predicted_graph = self.create_predicted_graph(parts, probabilities)

        return predicted_graph


    def _create_input_tensor(self, part_ids: list, part_id: int, size=2271) -> torch.Tensor:
        """
        Creates the input tensor for the model.
        :param part_ids: list of part ids
        :param part_id: part id
        :param size: size of the input tensor
        :return: input tensor
        """

        feature_graph = torch.zeros(size, dtype=torch.float)
        for pid in part_ids:
            feature_graph[pid] = 1

        feature_source_id = torch.zeros(size, dtype=torch.float)
        feature_source_id[part_id] = 1

        input_tensor = torch.cat([feature_graph, feature_source_id])
        
    
        return input_tensor
    

    def _predict(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output based on the input tensor.
        :param model: model
        :param input_tensor: input tensor
        :return: output tensor
        """

        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = model(input_tensor)
            return output.squeeze(0)  # Remove batch dimension from output
        

    def calculate_probabilities(self, model: nn.Module, part_ids) -> dict:
        """
        Calculates the probabilities for the given part ids.
        :param model: model
        :param part_ids: list of part ids
        :return: dictionary of dictionaries containing the probabilities
        """


        part_id_set = set(part_ids)

        # Create a matrix to store the probabilities (dictionary of dictionaries)
        probabilities = {part_id: {} for part_id in part_id_set}

        for part_id in part_id_set:
            input_tensor = self._create_input_tensor(part_ids, part_id)
            prediction = self._predict(model, input_tensor)
            # Only store the probabilities that are in the part_ids set
            for idx, prob in enumerate(prediction):
                if idx in part_ids:
                    probabilities[part_id][idx] = prob
        
        return probabilities
    
    def create_predicted_graph(self, parts: list, probabilities: dict) -> Graph:
        """
        Creates the predicted graph based on the given parts and probabilities.
        :param parts: list of parts
        :param probabilities: dictionary of dictionaries containing the probabilities
        :return: predicted graph
        """

        
        G = self._create_empty_graph_from_parts(parts)

        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 == node2:
                    continue
                part_id1 = int(node1.get_part_id())
                part_id2 = int(node2.get_part_id())
                prob = probabilities[part_id1].get(part_id2, 0)

                # Weight is the inverse of the probability, so that higher probabilities have lower weights
                weight = 1 - prob

                G.add_edge(node1, node2, weight=weight)

        T = nx.minimum_spanning_tree(G)

        predicted_custom_graph = Graph()

        for edge in T.edges():
            predicted_custom_graph.add_undirected_edge(edge[0], edge[1])

        return predicted_custom_graph
    
    def _create_empty_graph_from_parts(self, parts: Set[Part]) -> Graph:
        """
        Creates an empty graph from the given parts.
        :param parts: set of parts
        :return: graph
        """

        graph = nx.Graph()
        graph.add_nodes_from(parts)
        return graph