import pickle
import random
from typing import List
from graph import Graph

# Split the dataset into training and test sets

def split_dataset(graphs: List[Graph], train_ratio: float):
    random.shuffle(graphs)
    total = len(graphs)
    print(f"Total number of graphs: {total}")
    train_end = int(total * train_ratio)
    
    train_set = graphs[:train_end]
    test_set = graphs[train_end:]
    
    return train_set, test_set

if __name__ == '__main__':
    # Load the graphs
    with open('data/graphs.dat', 'rb') as file:
        graphs: List[Graph] = pickle.load(file)
    
    # Split the dataset
    train_set, test_set = split_dataset(graphs, train_ratio=0.8)
    
    # Save the splits
    with open('data/train_graphs.dat', 'wb') as file:
        pickle.dump(train_set, file)
    
    with open('data/test_graphs.dat', 'wb') as file:
        pickle.dump(test_set, file)
    
    print(f"Dataset split into {len(train_set)} training graphs and {len(test_set)} test graphs.")