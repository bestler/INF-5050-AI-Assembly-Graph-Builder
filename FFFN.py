from typing import Set
import pickle
from evaluation import MyPredictionModel
from graph import Graph
from part import Part
from node import Node
import torch
import torch.nn as nn
import networkx as nx

class FFFN(MyPredictionModel):
    
    def __init__(self):
        super().__init__()
        self.model = None
    
    
    def load_model(self, file_path: str) -> MyPredictionModel:
        # Load the model
        model = torch.load(file_path)
        
        self.model = model

        return model

    
    def predict_graph(self, parts: Set[Part]) -> Graph:

        family_ids = [int(part.get_family_id()) for part in parts]
        probabilities = self.calculate_probabilities(self.model, family_ids)
        predicted_graph = self.create_predicted_graph(parts, probabilities)

        return predicted_graph


    def _create_input_tensor(self, family_ids: list, family_id: int, size=96) -> torch.Tensor:
        

        feature_graph = torch.zeros(size, dtype=torch.float)
        for fid in family_ids:
            feature_graph[fid] = 1

        feature_source_id = torch.zeros(size, dtype=torch.float)
        feature_source_id[family_id] = 1

        input_tensor = torch.cat([feature_graph, feature_source_id])
        
    
        return input_tensor
    

    def calculate_probabilities(self, model: nn.Module, family_ids: list) -> dict:

        
        family_id_set = set(family_ids)

        # Create a matrix to store the probabilities (dictionary of dictionaries)
        probabilities = {family_id: {} for family_id in family_id_set}

        for family_id in family_id_set:
            input_tensor = self._create_input_tensor(family_ids, family_id)
            prediction = self._predict(model, input_tensor)
            # Only store the probabilities that are in the family_ids set
            for idx, prob in enumerate(prediction):
                if idx in family_ids and idx != family_id:
                    probabilities[family_id][idx] = prob
        
        return probabilities

    def _predict(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:

        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = model(input_tensor)
            return output.squeeze(0)  # Remove batch dimension from output
        

    ## Function to calaculate the probabilities
    def calculate_probabilities(self, model: nn.Module, family_ids) -> dict:

        family_id_set = set(family_ids)

        # Create a matrix to store the probabilities (dictionary of dictionaries)
        probabilities = {family_id: {} for family_id in family_id_set}

        for family_id in family_id_set:
            input_tensor = self._create_input_tensor(family_ids, family_id)
            prediction = self._predict(model, input_tensor)
            # Only store the probabilities that are in the family_ids set
            for idx, prob in enumerate(prediction):
                if idx in family_ids and idx != family_id:
                    probabilities[family_id][idx] = prob
        
        return probabilities
    
    # Function to create the predicted graph
    def create_predicted_graph(self, parts: list, probabilities: dict) -> Graph:
        
        G = self._create_empty_graph_from_parts(parts)

        for node1 in G.nodes:
            for node2 in G.nodes:
                if node1 == node2:
                    continue
                family_id1 = int(node1.get_family_id())
                family_id2 = int(node2.get_family_id())
                prob = probabilities[family_id1].get(family_id2, 0)

                # Weight is the inverse of the probability, so that higher probabilities have lower weights
                weight = 1 - prob

                G.add_edge(node1, node2, weight=weight)

        T = nx.minimum_spanning_tree(G)

        predicted_custom_graph = Graph()

        for edge in T.edges():
            predicted_custom_graph.add_undirected_edge(edge[0], edge[1])

        return predicted_custom_graph
    
    def _create_empty_graph_from_parts(self, parts: Set[Part]) -> Graph:
        graph = nx.Graph()
        graph.add_nodes_from(parts)
        return graph