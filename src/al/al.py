from collections import Counter
import heapq
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
from batchbald_redux import batchbald
import igraph as ig
import leidenalg as la

from src.al.utils import kmeans_graph, leiden_graph
from src.gnn.utils import highest_degree_unlbl_nodes, infer_model
from src.utils.constants import DEGREE_CSV
from src.utils.utils import save_to_csv
from torch_geometric.utils import degree

def rdn_sel(pyg_graph, al_batch):
    unlbl_ix = torch.where(pyg_graph.unlbl_mask)[0]
    selected_indices = torch.randint(0, len(unlbl_ix), (al_batch,))

    return selected_indices
    
def unc_sel(model, pyg_graph, al_batch):
    model.eval()
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    preds = preds.squeeze()

    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze()
    preds = preds[unlbl_ix]
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.topk(preds_diff, al_batch, largest=False)[1]

    return unlbl_ix[selected_indices]


def unc_kmeans_sel(model, pyg_graph, batch, **kwargs):
    model.eval()

    al = kwargs.get("al")
    al_batch = batch
    
    if al == "unc-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, al_batch, pyg_graph.unlbl_mask, **kwargs)
    elif al == "unc-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, al_batch, **kwargs)
    
    preds = model(pyg_graph.x, pyg_graph.edge_index)

    preds = preds.squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]
    
    preds_diff = torch.abs(preds[:,0] - preds[:,1])

    selected_indices = torch.zeros([al_batch], dtype=torch.int)
    
    preds_diff_copy = preds_diff.clone()
    for i in range(al_batch):
        preds_diff_copy[cluster_labels[unlbl_ix] != i] = torch.inf
        
        selected_indices[i] = torch.argmin(preds_diff_copy)

        preds_diff_copy[cluster_labels[unlbl_ix] != i] = preds_diff[cluster_labels[unlbl_ix] != i].clone()

    return selected_indices


def batchbald_sel(model, pyg_graph, al_batch, device, bald_iter=10):
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


def bald_sel(model, pyg_graph, al_batch, device, bald_iter=10):
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
    
def bald_cluster_sel(model, pyg_graph, batch, **kwargs):
    device = kwargs.get("device")
    num_test = kwargs.get("num_test_inference_run")
    al = kwargs.get("al")
    threshold_cluster = kwargs.get("threshold_cluster")

    model.eval()

    preds = infer_model(model, pyg_graph, num_test, **kwargs)
    
    if al == "bald-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch, pyg_graph.unlbl_mask)
    elif al == "bald-leiden" or al == "bald-leiden-aug":
        cluster_labels = leiden_graph(pyg_graph)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]
    elif al == "bald-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]

    selected_indices = np.full([batch], -1, dtype=np.int32)

    preds = preds.squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]

    qtde_clusters_threshold = 0
    for i in range(batch):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        num_samples = preds_unlbl_hd.shape[0]
        if num_samples > threshold_cluster:
            qtde_clusters_threshold += 1
    
    if qtde_clusters_threshold == 0:
        qtde_clusters_threshold = batch
        threshold_cluster = 0
        
    qtde_per_cluster = np.full(qtde_clusters_threshold, batch // qtde_clusters_threshold, dtype=np.int32)
    qtde_per_cluster[:batch % qtde_clusters_threshold] += 1

    print(threshold_cluster, batch, qtde_clusters_threshold, qtde_per_cluster)

    j = 0
    start_ix = 0
    for i in range(batch):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        num_samples = preds_unlbl_hd.shape[0]
        if num_samples > threshold_cluster:
            qtde = min(qtde_per_cluster[j], num_samples)
            end_ix = start_ix + qtde

            if preds_unlbl_hd.dim() == 3:
                candidate_batch = batchbald.get_bald_batch(
                    preds_unlbl_hd, qtde, dtype=torch.double, device=device
                )
                selected_indices[start_ix:end_ix] = unlbl_ix[cluster_indices[candidate_batch.indices]]
            else:
                selected_indices[start_ix:end_ix] = unlbl_ix[cluster_indices].squeeze()

            start_ix = end_ix
            
            j += 1

    return selected_indices

def batchbald_cluster_sel(model, pyg_graph, batch, **kwargs):
    device = kwargs.get("device")
    num_test = kwargs.get("num_test_inference_run")
    al = kwargs.get("al")
    threshold_cluster = kwargs.get("threshold_cluster")
    n_clusters = batch

    model.eval()

    preds = infer_model(model, pyg_graph, num_test, **kwargs)
    
    if al == "batchbald-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, n_clusters, pyg_graph.unlbl_mask)
    elif al == "batchbald-leiden" or "batchbald-leiden-aug":
        cluster_labels = leiden_graph(pyg_graph)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]
    elif al == "batchbald-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, n_clusters)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]

    selected_indices = np.zeros([batch], dtype=np.int32)

    preds = preds.detach().squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]

    qtde_clusters_threshold = 0
    for i in range(batch):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        num_samples = preds_unlbl_hd.shape[0]
        if num_samples > threshold_cluster:
            qtde_clusters_threshold += 1
    
    if qtde_clusters_threshold == 0:
        qtde_clusters_threshold = batch
        threshold_cluster = 0
        
    qtde_per_cluster = np.full(qtde_clusters_threshold, batch // qtde_clusters_threshold, dtype=np.int32)
    qtde_per_cluster[:batch % qtde_clusters_threshold] += 1

    print(threshold_cluster, batch, qtde_clusters_threshold, qtde_per_cluster)

    start_index = 0
    j = 0
    for i in range(n_clusters):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        num_samples = preds_unlbl_hd.shape[0]
        if num_samples > threshold_cluster:
            qtde = min(qtde_per_cluster[j], num_samples)

            if preds_unlbl_hd.dim() == 3 and num_samples >= qtde * 2:
                candidate_batch = batchbald.get_batchbald_batch(
                    preds_unlbl_hd, qtde, num_samples, dtype=torch.double, device=device
                )
                end_index = start_index + qtde
                selected_indices[start_index:end_index] = unlbl_ix[cluster_indices[candidate_batch.indices]]

            elif preds_unlbl_hd.dim() == 3 and qtde < num_samples < qtde * 2:
                end_index = start_index + qtde
                candidate_batch = batchbald.get_bald_batch(
                    preds_unlbl_hd, qtde, dtype=torch.double, device=device
                )
                selected_indices[start_index:end_index] = unlbl_ix[cluster_indices[candidate_batch.indices]]

            elif num_samples > 0:
                end_index = start_index + num_samples
                selected_indices[start_index:end_index] = unlbl_ix[cluster_indices]
            
            start_index = end_index
            j += 1
    
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
    
    print(al_batch_pseudo_train)
        
    if al == "random":
        selected_indices = rdn_sel(pyg_graph_train, al_batch_pseudo_train)
        
    elif al == "unc":
        selected_indices = unc_sel(model, pyg_graph_train, al_batch_pseudo_train)
        
    elif al == "unc-kmeans" or al == "unc-kmeans-aug":
        selected_indices = unc_kmeans_sel(model, pyg_graph_train, al_batch_pseudo_train, **kwargs)

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

    elif al == "bald-kmeans" or al == "bald-leiden" or al == "bald-kmeans-aug" or al == "bald-leiden-aug":
        selected_indices = bald_cluster_sel(model, pyg_graph_train, al_batch_pseudo_train, **kwargs)
        selected_indices = selected_indices.squeeze()
    
    elif al == "batchbald-kmeans" or al == "batchbald-leiden" or al == "batchbald-kmeans-aug" or al == "batchbald-leiden-aug":
        selected_indices = batchbald_cluster_sel(model, pyg_graph_train, al_batch_pseudo_train, **kwargs)
        selected_indices = selected_indices.squeeze()

    elif al == "kmeans":
        selected_indices = cluster_sel(model, pyg_graph_train, al_batch_pseudo_train, **kwargs)
        selected_indices = selected_indices.squeeze()

    degree_report = dict()
    degree_report["exp"] = kwargs.get("exp_id")
    degree_report["mode"] = "active-learning"
    degree_report["labeled_size"] = kwargs.get("lbl_num")
    degree_report["set_id"] = kwargs.get("set_id")
    degree_report["run_id"] = kwargs.get("run_id")

    deg = degree(pyg_graph_train.edge_index[0], num_nodes=pyg_graph_train.num_nodes)
    degree_report["mean_degree"] = torch.mean(deg[selected_indices]).cpu().numpy()
    # degree_report["mean_bw"] = torch.mean(pyg_graph_train.betweenness[selected_indices]).cpu().numpy()

    csv_file = DEGREE_CSV.format(kwargs.get("event"), kwargs.get("exp_group"))

    # Save degree_report to CSV file
    save_to_csv(degree_report, csv_file)

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
