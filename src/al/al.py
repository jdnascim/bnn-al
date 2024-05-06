from sklearn.cluster import KMeans
import torch
from batchbald_redux import batchbald

from src.utils.gnn import highest_degree_unlbl_nodes

def rdn_sel(pyg_graph, al_batch):
    unlbl_ix = torch.where(pyg_graph.unlbl_mask)[0]
    selected_indices = torch.randint(0, len(unlbl_ix), (al_batch,))

    return selected_indices
    
def unc_sel(model, pyg_graph, al_batch):
    model.eval()
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    preds = preds.squeeze()
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.topk(preds_diff, al_batch, largest=False)[1]

    return selected_indices

def unc_kmeans_sel(model, pyg_graph, al_batch):
    model.eval()
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    kmeans = KMeans(n_clusters=al_batch)

    feature_matrix = pyg_graph.x.detach().cpu().numpy()
    kmeans.fit(feature_matrix)
    cluster_labels = kmeans.labels_

    preds = preds.squeeze()
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.zeros([al_batch], dtype=torch.int)
    
    preds_diff_copy = preds_diff.clone()
    for i in range(al_batch):
        preds_diff_copy[cluster_labels != i] = torch.inf
        
        selected_indices[i] = torch.argmin(preds_diff_copy)

        preds_diff_copy[cluster_labels != i] = preds_diff[cluster_labels != i].clone()

    return selected_indices


def bald_base(model, pyg_graph, al_batch, bald_iter=10):
    model.eval()

    preds_mt = torch.zeros([bald_iter, pyg_graph.x.shape[0], 2])
    
    for i in range(bald_iter):
        preds_mt[i] = model(pyg_graph.x, pyg_graph.edge_index, dropout_infer=True)
    
    preds_std = torch.std(preds_mt, 0)

    # to consider only unlabeled data from the event
    preds_std[pyg_graph.event_unlbl_mask == False] = -1 * torch.inf

    selected_indices = torch.topk(preds_std[:,1],k=al_batch)[1]

    return selected_indices


def bald_kmeans(model, pyg_graph, al_batch, bald_iter=10):
    model.eval()
    
    preds_mt = torch.zeros([bald_iter, pyg_graph.x.shape[0], 2])
    
    for i in range(bald_iter):
        preds_mt[i] = model(pyg_graph.x, pyg_graph.edge_index, dropout_infer=True)

    preds_std = torch.std(preds_mt, 0)

    kmeans = KMeans(n_clusters=al_batch)

    feature_matrix = pyg_graph.x.detach().cpu().numpy()
    kmeans.fit(feature_matrix)
    cluster_labels = kmeans.labels_

    # to consider only unlabeled data from the event
    preds_std[pyg_graph.event_unlbl_mask == False] = -1 * torch.inf

    selected_indices = torch.zeros([al_batch], dtype=torch.int)
    
    preds_std_copy = preds_std.clone()
    for i in range(al_batch):
        preds_std_copy[cluster_labels != i] = -1 * torch.inf
        
        selected_indices[i] = torch.argmax(preds_std_copy[:,1])

        preds_std_copy[cluster_labels != i] = preds_std[cluster_labels != i].clone()

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
