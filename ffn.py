import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import pickle
from evaluation import MyPredictionModel, evaluate
from torch.nn.utils.rnn import pad_sequence

# -------------------------
# 📝 Data Preparation
# -------------------------
class GraphDataset(Dataset):
    def __init__(self, data_path):
        # Load dataset
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.data[idx]
        
        # Extract nodes from the graph
        if not hasattr(graph, '_Graph__nodes'):
            raise ValueError(f"Graph object at index {idx} does not have '_Graph__nodes' attribute.")

        parts = set(node._Node__part for node in graph._Graph__nodes)
        
        part_ids = torch.tensor([int(part._Part__part_id) for part in parts], dtype=torch.long)
        family_ids = torch.tensor([int(part._Part__family_id) for part in parts], dtype=torch.long)
        
        return part_ids, family_ids, graph

# One-hot encoding for part IDs
def one_hot_encode(parts, num_classes):
    return F.one_hot(parts, num_classes=num_classes).float()

def collate_fn(batch):
    """
    Custom collate function to handle variable graph sizes in batches.
    """
    part_ids_batch = []
    family_ids_batch = []
    edge_labels_batch = []
    graphs = []
    
    for part_ids, family_ids, graph in batch:
        part_ids_batch.append(part_ids)
        family_ids_batch.append(family_ids)
        
        # Create edge matrix
        num_parts = len(part_ids)
        edge_labels = torch.zeros((num_parts, num_parts))
        for i, node in enumerate(graph._Graph__nodes):
            for neighbor in graph._Graph__edges.get(node, []):
                edge_labels[i][neighbor._Node__id] = 1
        
        edge_labels_batch.append(edge_labels)
        graphs.append(graph)
    
    # Pad part_ids and family_ids to the max length in the batch
    part_ids_padded = pad_sequence(part_ids_batch, batch_first=True, padding_value=-1)
    family_ids_padded = pad_sequence(family_ids_batch, batch_first=True, padding_value=-1)
    
    # Pad edge matrices
    max_nodes = max(edge.shape[0] for edge in edge_labels_batch)
    edge_labels_padded = torch.zeros((len(batch), max_nodes, max_nodes))
    for i, edge_matrix in enumerate(edge_labels_batch):
        edge_labels_padded[i, :edge_matrix.shape[0], :edge_matrix.shape[1]] = edge_matrix
    
    return part_ids_padded, family_ids_padded, edge_labels_padded, graphs


# -------------------------
# 🧠 Model Definition
# -------------------------
class GraphPredictionModel(MyPredictionModel, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_out = nn.Linear(hidden_dim, output_dim)  # Predict edges

    def forward(self, part_features):
        x = F.relu(self.fc1(part_features))
        x = F.relu(self.fc2(x))
        edge_probs = torch.sigmoid(self.edge_out(x))
        return edge_probs

    def predict_graph(self, parts):
        # Predict edges for the given set of parts
        part_features = one_hot_encode(torch.tensor([p.part_id for p in parts]), num_classes=1089)
        edge_probs = self.forward(part_features)
        predicted_edges = edge_probs > 0.5  # Binary threshold for edges
        # Build a graph from predicted edges
        graph = self.build_graph(parts, predicted_edges)
        return graph

    def build_graph(self, parts, predicted_edges):
        from graph import Graph, Node
        graph = Graph()
        nodes = [Node(i, part) for i, part in enumerate(parts)]
        for i, node in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if predicted_edges[i][j]:
                    graph.add_edge(node, nodes[j])
        return graph


# -------------------------
# 📊 Training Function
# -------------------------
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for part_ids_padded, family_ids_padded, edge_labels_padded, graphs in train_loader:
            # Mask to ignore padding in loss calculation
            valid_mask = (part_ids_padded != -1)

            # Replace -1 padding with 0 (or any valid index, it won't affect the masked areas)
            part_ids_padded = torch.where(valid_mask, part_ids_padded, torch.tensor(0, dtype=torch.long))

            # Flatten and process batch
            batch_size, max_nodes = edge_labels_padded.shape[0], edge_labels_padded.shape[1]
            part_features = F.one_hot(part_ids_padded, num_classes=1089).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            edge_preds = model(part_features.view(batch_size * max_nodes, -1))
            edge_preds = edge_preds.view(batch_size, max_nodes, -1)
            
            # Apply mask to focus only on valid nodes
            edge_preds = edge_preds * valid_mask.unsqueeze(-1)
            edge_labels_padded = edge_labels_padded * valid_mask.unsqueeze(-1)
            
            # Compute loss
            loss = criterion(edge_preds, edge_labels_padded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for part_ids, family_ids, graph in val_loader:
                part_features = one_hot_encode(part_ids, num_classes=1089)
                edge_labels = torch.zeros((len(part_ids), len(part_ids)))
                for i, node in enumerate(graph._Graph__nodes):
                    for neighbor in graph._Graph__edges[node]:
                        edge_labels[i][neighbor._Node__id] = 1
                
                edge_preds = model(part_features)
                loss = criterion(edge_preds, edge_labels)
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


# -------------------------
# 🚀 Main Execution
# -------------------------
def main():
    # Hyperparameters
    input_dim = 1089  # One-hot encoding for 1089 parts
    hidden_dim = 256
    output_dim = 1089  # Predict connections for each node
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001

    # Data Loaders
    train_dataset = GraphDataset('data/train_graphs.dat')
    val_dataset = GraphDataset('data/val_graphs.dat')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Model, Loss, Optimizer
    model = GraphPredictionModel(input_dim, hidden_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Save Model
    torch.save(model.state_dict(), 'graph_prediction_model.pth')

    # Evaluate on Test Set
    test_dataset = GraphDataset('data/test_graphs.bat')
    score = evaluate(model, test_dataset)
    print(f"Final Evaluation Score: {score}")


if __name__ == '__main__':
    main()