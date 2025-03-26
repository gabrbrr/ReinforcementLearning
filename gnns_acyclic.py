import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from imblearn.over_sampling import RandomOverSampler

def normalize_adj(adj):
    adj = adj.numpy()
    N = adj.shape[0]
    adj = adj + np.eye(N)
    D = np.sum(adj, axis=0)
    D_hat = np.diag(D**-0.5)
    out = np.dot(D_hat, adj).dot(D_hat)
    return torch.tensor(out, dtype=torch.float32)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, A, X):
        X = torch.mm(A, X)
        return torch.sigmoid(self.linear(X))

class GNNIsAcyclic(nn.Module):
    def __init__(self, input_dim, gcn_dims=[8, 16], output_dim=2):
        super(GNNIsAcyclic, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(input_dim, gcn_dims[0]))
        self.gcn_layers.append(GCNLayer(gcn_dims[0], gcn_dims[1]))
        self.classifier = nn.Linear(gcn_dims[-1], output_dim)
        self.soft=nn.Softmax()

    def forward(self, X, A):
        for gcn in self.gcn_layers:
            X = gcn(A, X)
        graph_embedding = torch.mean(X, dim=0)  
        logits = self.classifier(graph_embedding)
        probs =  self.soft(logits)
        return logits, probs

def oversample_data(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(np.array(X, dtype=object).reshape(-1, 1), y)
    return [x[0] for x in X_resampled], y_resampled

    
def process_graph(graph):
    node_features = np.array([graph.degree(node) for node in graph.nodes]).reshape(-1, 1)  # Node degrees
    adj_matrix = nx.to_numpy_array(graph)
    return torch.tensor(node_features, dtype=torch.float32), torch.tensor(adj_matrix, dtype=torch.float32)

def generate_balanced_dataset():
    datasets = []
    labels = []
    
    for i in range(2, 9):
        for j in range(2, 9):
            datasets.append(nx.grid_2d_graph(i, j))
            labels.append(0)
    
    for i in range(3, 30):  
        datasets.append(nx.cycle_graph(i))
        labels.append(0)
    
    for i in range(15): 
        datasets.append(nx.cycle_graph(3))
        labels.append(0)
    
    for i in range(2, 30):
        datasets.append(nx.wheel_graph(i))
        labels.append(0)
    
    for i in range(2, 20):
        datasets.append(nx.circular_ladder_graph(i))
        labels.append(0)
    
    for i in range(2, 65):
        datasets.append(nx.star_graph(i))
        labels.append(1)
    
    g = nx.balanced_tree(2, 5)
    datasets.append(g)
    labels.append(1)
    for i in range(62, 2, -1):
        g.remove_node(i)
        datasets.append(g.copy())  
        labels.append(1)
    
    for i in range(3, 65):
        datasets.append(nx.path_graph(i))
        labels.append(1)
    
    for i in range(3, 5):
        for j in range(5, 30): 
            datasets.append(nx.full_rary_tree(i, j))
            labels.append(1)
    
    
    return datasets,labels  



if __name__ == "__main__":
    		
    datasets,labels=generate_balanced_dataset()
    datasets,labels=oversample_data(datasets,labels)
    print(labels.count(1),len(labels))
    
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(datasets, labels, test_size=0.1, random_state=42)
    input_dim = 1 
    model = GNNIsAcyclic(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10000
    batch_size = 32
    best_val_loss = float('inf')  

    dataset_size = len(train_graphs)
    print("Training Dataset size:", dataset_size)
    print("Validation Dataset size:", len(val_graphs))

    for epoch in range(epochs):
        permutation = np.random.permutation(dataset_size)
        epoch_loss = 0
        train_correct = 0

        model.train()
        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            batch_indices = permutation[i:i + batch_size]
            batch_loss = 0

            for idx in batch_indices:
                graph = train_graphs[idx]
                label = train_labels[idx]

                X, A = process_graph(graph)
                A = normalize_adj(A)
                logits, _ = model(X, A)
                loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long))
                batch_loss += loss.item()

                _, predicted = torch.max(logits, dim=0)
                train_correct += (predicted.item() == label)
                loss.backward()

            optimizer.step()
            epoch_loss += batch_loss / len(batch_indices)

        train_acc = train_correct / dataset_size

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for idx in range(len(val_graphs)):
                graph = val_graphs[idx]
                label = val_labels[idx]

                X, A = process_graph(graph)
                A = normalize_adj(A)
                logits,_ = model(X, A)
                loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long))
                val_loss += loss.item()

                _, predicted = torch.max(logits, dim=0)
                val_correct += (predicted.item() == label)

        val_loss /= len(val_graphs)
        val_acc = val_correct / len(val_graphs)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            print("Saving best model...")
            state = {
                'net': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(state, "./checkpoint/gnn_is_acyclic_best.pth")
            best_val_loss = val_loss

    print("Best Validation Loss:", best_val_loss)

    input_dim = 1  
    model = GNNIsAcyclic(input_dim)
    checkpoint_path = "./checkpoint/gnn_is_acyclic_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found!")

    



