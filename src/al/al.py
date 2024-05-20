from collections import Counter
import heapq
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
from batchbald_redux import batchbald
import igraph as ig
import leidenalg as la

from src.gnn.utils import highest_degree_unlbl_nodes

def rdn_sel(pyg_graph, al_batch):
    unlbl_ix = torch.where(pyg_graph.unlbl_mask)[0]
    selected_indices = torch.randint(0, len(unlbl_ix), (al_batch,))

    return selected_indices
    
def unc_sel(model, pyg_graph, al_batch):
    model.eval()
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    preds = preds.squeeze()

    unlbl_ix = torch.where(pyg_graph.unlbl_mask == True)
    preds = preds[unlbl_ix]
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.topk(preds_diff, al_batch, largest=False)[1]

    return unlbl_ix[selected_indices]


def unc_kmeans_sel(model, pyg_graph, al_batch):
    model.eval()
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    kmeans = KMeans(n_clusters=al_batch)

    feature_matrix = pyg_graph.x.detach().cpu().numpy()
    kmeans.fit(feature_matrix)
    cluster_labels = kmeans.labels_

    preds = preds.squeeze()
    unlbl_ix = torch.where(pyg_graph.unlbl_mask == True)
    preds = preds[unlbl_ix]
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.zeros([al_batch], dtype=torch.int)
    
    preds_diff_copy = preds_diff.clone()
    for i in range(al_batch):
        preds_diff_copy[cluster_labels != i] = torch.inf
        
        selected_indices[i] = torch.argmin(preds_diff_copy)

        preds_diff_copy[cluster_labels != i] = preds_diff[cluster_labels != i].clone()

    return selected_indices


def batchbald_sel(model, pyg_graph, al_batch, device, bald_iter=100):
    model.eval()
    
    preds_mt= model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    preds_unlbl = preds_mt[pyg_graph.unlbl_mask]
    original_index = torch.argwhere(pyg_graph.unlbl_mask == True)

    candidate_batch = batchbald.get_batchbald_batch(
        preds_unlbl, al_batch, al_batch, dtype=torch.double, device=device
    )

    selected_indices = original_index[candidate_batch.indices]

    return selected_indices


def bald_sel(model, pyg_graph, al_batch, device, bald_iter=100):
    model.eval()
    
    preds_mt= model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    preds_unlbl = preds_mt[pyg_graph.unlbl_mask]
    original_index = torch.argwhere(pyg_graph.unlbl_mask == True)

    candidate_batch = batchbald.get_bald_batch(
        preds_unlbl, al_batch, dtype=torch.double, device=device
    )

    selected_indices = original_index[candidate_batch.indices]

    return selected_indices


def batchbald_deg_sel(model, pyg_graph, al_batch, device, k_batch=4, bald_iter=100):
    model.eval()
    
    preds_mt = model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    hd_unlbl_nodes = highest_degree_unlbl_nodes(pyg_graph, k_batch*al_batch)

    preds_unlbl_hd = preds_mt[hd_unlbl_nodes]

    candidate_batch = batchbald.get_batchbald_batch(
        preds_unlbl_hd, al_batch, al_batch, dtype=torch.double, device=device
    )

    selected_indices = hd_unlbl_nodes[candidate_batch.indices]

    return selected_indices

    
def bald_deg_sel(model, pyg_graph, al_batch, device, k=64, bald_iter=100):
    model.eval()
    
    preds_mt= model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    hd_unlbl_nodes = highest_degree_unlbl_nodes(pyg_graph, k)

    preds_unlbl_hd = preds_mt[hd_unlbl_nodes]

    candidate_batch = batchbald.get_bald_batch(
        preds_unlbl_hd, al_batch, dtype=torch.double, device=device
    )

    selected_indices = hd_unlbl_nodes[candidate_batch.indices]

    return selected_indices

def batchbald_leiden_sel(model, pyg_graph, al_batch, device, k_batch=4, bald_iter=100):
    model.eval()
    
    preds_mt = model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    edge_index = pyg_graph.edge_index
    num_nodes = pyg_graph.num_nodes
    
    # Create igraph.Graph
    G = ig.Graph(n=num_nodes)
    G.add_edges(edge_index.T.tolist())  # Convert edge_index to list of edges

    partition = la.find_partition(G, la.ModularityVertexPartition);

    hd_unlbl_nodes = highest_degree_unlbl_nodes(pyg_graph, k_batch*al_batch)

    preds_unlbl_hd = preds_mt[hd_unlbl_nodes]

    candidate_batch = batchbald.get_batchbald_batch(
        preds_unlbl_hd, al_batch, al_batch, dtype=torch.double, device=device
    )

    selected_indices = hd_unlbl_nodes[candidate_batch.indices]

    return selected_indices
    
    
def al_update(model, pyg_graph_train, pyg_graph_dev, **kwargs):
    al = kwargs.get("al")
    random_pseudo_val = kwargs.get("random_pseudo_val")
    lbl_train_frac = kwargs.get("lbl_train_frac")
    al_batch = kwargs.get("al_batch")
    device = kwargs.get("device")
    set_id = kwargs.get("set_id")

    if random_pseudo_val:
        al_batch_pseudo_train = int(al_batch * lbl_train_frac)
        al_batch_pseudo_val = al_batch - al_batch_pseudo_train
    else:
        al_batch_pseudo_train = al_batch
        
    if al == "random":
        selected_indices = rdn_sel(pyg_graph_train, al_batch_pseudo_train)
        
    elif al == "unc":
        selected_indices = unc_sel(model, pyg_graph_train, al_batch_pseudo_train)
        
    elif al == "unc-kmeans":
        selected_indices = unc_kmeans_sel(model, pyg_graph_train, al_batch_pseudo_train)

    elif al == "batchbald":
        selected_indices = batchbald_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
        selected_indices = selected_indices.squeeze()

    elif al == "bald":
        selected_indices = bald_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
        selected_indices = selected_indices.squeeze()

    elif al == "batchbald-degree":
        selected_indices = batchbald_deg_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
        selected_indices = selected_indices.squeeze()

    elif al == "bald-degree":
        selected_indices = bald_deg_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
        selected_indices = selected_indices.squeeze()

    pyg_graph_train.labeled_mask[selected_indices] = True
    pyg_graph_train.unlbl_mask[selected_indices] = False
    pyg_graph_dev.labeled_mask[selected_indices] = True
    pyg_graph_dev.unlbl_mask[selected_indices] = False

    if random_pseudo_val:
        selected_indices_pseudo_val = rdn_sel(pyg_graph_train, al_batch_pseudo_val)
        pyg_graph_train.pseudo_train_mask[selected_indices] = True
        pyg_graph_train.pseudo_val_mask[selected_indices_pseudo_val] = True
        pyg_graph_train.labeled_mask[selected_indices_pseudo_val] = True
        pyg_graph_train.unlbl_mask[selected_indices_pseudo_val] = False
        pyg_graph_dev.pseudo_train_mask[selected_indices] = True
        pyg_graph_dev.pseudo_val_mask[selected_indices_pseudo_val] = True
        pyg_graph_dev.labeled_mask[selected_indices_pseudo_val] = True
        pyg_graph_dev.unlbl_mask[selected_indices_pseudo_val] = False

    else:
        pseudo_train, pseudo_val = train_test_split(selected_indices, train_size=lbl_train_frac, random_state=set_id)
        pyg_graph_train.pseudo_train_mask[pseudo_train] = True
        pyg_graph_train.pseudo_val_mask[pseudo_val] = True
        pyg_graph_dev.pseudo_train_mask[pseudo_train] = True
        pyg_graph_dev.pseudo_val_mask[pseudo_val] = True

    return pyg_graph_train, pyg_graph_dev
