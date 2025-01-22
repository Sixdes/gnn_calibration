import torch
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import random
import pickle
import copy
import numpy as np
import scipy.sparse as sp
from pathlib import Path
# import os
# import sys
# sys.path.append("../")


def modify_del_graph(data, delete_ratio=0.1):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        delete_ratio (float): The ratio of edges to delete based on label inconsistency.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()
    
    # Randomly delete edges with hete labels
    edges_to_delete = []
    for u, v in edges:
        rand = random.random()
        if labels[u] != labels[v] and rand <= delete_ratio:
            edges_to_delete.append([u, v])

    for element in edges_to_delete:
        edges.remove(element)
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index

def modify_add_graph(data, add_num=0):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_num (float): The num of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []
    idx = 0
    while(idx <= add_num):
        u, v = random.sample(range(num_nodes), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
            idx = idx + 1
    print(f'add the num of edges: {idx-1}')
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    return new_edge_index

def count_hete_edges(edge_index, labels):
    """
    Count the number of edges with hete node labels in the graph data.

    Args:
        data (Data): The graph data object from torch_geometric.
    
    Returns:
        int: The number of edges with hete node labels.
    """
    
    hete_edges_count = 0
    hete_edges_list = []
    
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        
        if labels[src] != labels[tgt]:
            hete_edges_count += 1
            hete_edges_list.append([src, tgt])
    
    return hete_edges_count, hete_edges_list

def cal_hete_edge_rate(new_edge_index, hete_edges_list):
    mod_hete_edges = 0
    new_edges = new_edge_index.t().tolist()
    
    for edge_idx in hete_edges_list:
        src, tgt = edge_idx
        # Check if the edge still exists in the modified graph
        if [src, tgt] in new_edges:
            mod_hete_edges += 1
    
    modification_rate = 1 - mod_hete_edges / len(hete_edges_list) if len(hete_edges_list) > 0 else 0.0
    
    return modification_rate

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):

    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        print(f'the modify-delete edges: {n_edges} with num {n_remove} ratio {remove_pct / 100}') 
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        # n_add = int(n_edges * add_pct / 100)
        n_add = add_pct
        print(f'the modify-add edges: {n_edges} with num {n_add}') 
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    # adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    # adj_pred = adj_pred + adj_pred.T
    edges_pred = edges_pred.tolist()
    reversed_edges = [[edge[1], edge[0]] for edge in edges_pred]
    edges_pred += reversed_edges
    edges = torch.tensor(edges_pred, dtype=torch.long).t()
    edge_index = edges.contiguous()
    
    return edge_index

def modify_add_graph_ratio(data, add_ratio=0.1):
    """
    Modify the graph by randomly deleting and adding edges based on node labels.

    Args:
        data (Data): The graph data object from torch_geometric.
        add_ratio (float): The ratio of edges to add between nodes with the same label.
    
    Returns:
        Data: The modified graph data with updated edges.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    dtype = edge_index.dtype
    
    # Convert edge_index to a list of edges
    edges = edge_index.t().tolist()
    labels = data.y.cpu().numpy()

    # Randomly add edges between nodes with the same label
    num_nodes = data.num_nodes
    edges_to_add = []

    # num_edges_to_add = int(num_nodes * (num_nodes - 1) * add_ratio
    num_edges_to_add = int(num_edges * add_ratio / 2)
    
    for _ in range(num_edges_to_add):  
        u, v = random.sample(range(num_nodes), 2)
        if labels[u] == labels[v] and [u, v] not in edges and [v, u] not in edges:
            edges_to_add.append([u, v])
            edges_to_add.append([v, u])
            edges.append([u, v])
            edges.append([v, u])
    
    # Convert edges back to edge_index format
    new_edge_index = torch.tensor(edges, dtype=dtype).t().contiguous().to(device)
    
    # Create a new Data object with the modified edge_index
    # modified_data = Data(x=data.x, edge_index=new_edge_index, y=data.y).to(device)
    
    return new_edge_index


if __name__ == '__main__':
    # Example usage
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a random graph for demonstration
    dataset_cora = Planetoid('/root/ytx/calibration-gnn/GATS/data/', 'Cora', transform=T.NormalizeFeatures())
    
    data = dataset_cora[0].to(device)
    labels = data.y.cpu().numpy()
    edge_index = data.edge_index
    edge = edge_index.t().tolist()

    # Modify the graph
    delete_edge_index = modify_del_graph(data, delete_ratio=0.3)

    count1,cout1_list = count_hete_edges(edge_index, labels)
    count2,count2_list = count_hete_edges(delete_edge_index, labels)
    acc_heto = 1 - count2/count1

    acc_heto2 = cal_hete_edge_rate(delete_edge_index, cout1_list)

    add_edge_index = modify_add_graph(data, add_num=200)

    print("Original edges:")
    print(data.edge_index)
    print("Modified delete edges:")
    print(delete_edge_index)
    print("Modified add edges:")
    print(add_edge_index)