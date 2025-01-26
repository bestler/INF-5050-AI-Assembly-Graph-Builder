import pickle
from typing import List, Set, Tuple, Dict
from abc import ABC, abstractmethod
from itertools import permutations
import numpy as np
from sklearn.model_selection import train_test_split
from graph import Graph
from part import Part
from prediction_classes.naive_predictor import SimpleEdgeFrequencyModel
import torch
import torch.nn as nn
from prediction_classes.family_predictor import FamilyPredictor
from prediction_classes.part_predictor import PartPredictor
from ffn import FFNModel
import numpy as np
import time


class MyPredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @abstractmethod
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate()`.
        :param parts: set of parts to form up an assembly (i.e. a graph)
        :return: graph
        """
        # TODO: implement this method
        ...


def load_model(file_path: str) -> MyPredictionModel:
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def evaluate(model: MyPredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> dict:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    num_graphs = len(data_set)
    correct_predicted_graphs = 0

    sum_correct_edges = 0
    edges_counter = 0

    test = 0

    for input_parts, target_graph in data_set:
        # print(test)
        # test += 1
        #print(len(input_parts))
        predicted_graph = model.predict_graph(input_parts)

        # We prepared a simple evaluation metric `edge_accuracy()`for you
        # Think of other suitable metrics for this task and evaluate your model on them!
        # FYI: maybe some more evaluation metrics will be used in final evaluation
        edges_counter += len(input_parts) * len(input_parts)
        sum_correct_edges += edge_accuracy(predicted_graph, target_graph)

        ## Exact match metric
        correct_predicted_graphs += exact_match(predicted_graph, target_graph)

    total_edge_accuracy = sum_correct_edges / edges_counter * 100

    exact_match_score = correct_predicted_graphs / num_graphs * 100

    # return value in percent
    return {'edge_accuracy': total_edge_accuracy, 'exact_match': exact_match_score}


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    A simple evaluation metric: Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(predicted_graph.get_parts())

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score

def exact_match(predicted_graph: Graph, target_graph: Graph) -> bool:
    """
    Returns True if the predicted graph is exactly the same as the target graph.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'

    return predicted_graph == target_graph


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all([len(perm) == len(parts) for perm in full_perms]), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Example code for evaluation

if __name__ == '__main__':

    # Prüfen, ob die Daten konsistent geladen werden
    with open('data/graphs.dat', 'rb') as file:
        graphs: List[Graph] = pickle.load(file)


    _, test_graphs = train_test_split(graphs, test_size=0.15, random_state=65)

    # Uncomment here the model you want to evaluate. Unser Hauptansatz ist der PartPredictor

    # naive_predictor = load_model('edge_prediction_models/simple_edge_frequency_model.pkl')


    # family_predictor = FamilyPredictor()
    # family_predictor.load_model('edge_prediction_models/family_edge_prediction_model.pth')

    part_predictor = PartPredictor()
    part_predictor.load_model('edge_prediction_models/part_edge_prediction_model.pth')


    instances = [[graph.get_parts(), graph] for graph in test_graphs]


    start = time.time()
    eval_results = evaluate(part_predictor, instances)
    end = time.time()

    print(f'Evaluation results: {eval_results}')
    
    print(f'Evaluation time: {end - start:.2f}s')