import numpy as np
from sklearn.model_selection import train_test_split
import torch
from src.gnn.utils import infer_model
from src.al.utils import kmeans_graph, leiden_graph
from src.batchbald import batchbald_inverted
from batchbald_redux import batchbald
from torch_geometric.utils import degree

from src.utils.constants import DEGREE_CSV
from src.utils.utils import save_to_csv

def bald_cluster_pl(model, pyg_graph, batch, **kwargs):
    device = kwargs.get("device")
    num_test = kwargs.get("num_test_inference_run")
    pl = kwargs.get("pl")

    model.eval()

    preds = infer_model(model, pyg_graph, num_test, **kwargs)
    
    if pl == "bald-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch, pyg_graph.unlbl_mask, **kwargs)
    elif pl == "bald-leiden":
        cluster_labels, _ = leiden_graph(pyg_graph, batch)
    elif pl == "bald-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch, **kwargs)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]

    selected_indices = np.full([batch], -1, dtype=np.int32)

    preds = preds.squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]

    for i in range(batch):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        if preds_unlbl_hd.dim() == 3 and preds_unlbl_hd.shape[0] > 0:
            candidate_batch = batchbald.get_bald_batch(
                preds_unlbl_hd, preds_unlbl_hd.shape[0], dtype=torch.double, device=device
            )
            selected_indices[i] = unlbl_ix[cluster_indices[candidate_batch.indices[-1]]]
        elif preds_unlbl_hd.shape[0] > 0:
            selected_indices[i] = unlbl_ix[cluster_indices]

    return selected_indices


def batchbald_cluster_pl(model, pyg_graph, batch, **kwargs):
    device = kwargs.get("device")
    num_test = kwargs.get("num_test_inference_run")
    pl = kwargs.get("pl")
    threshold_cluster = kwargs.get("threshold_cluster")
    n_clusters = batch

    model.eval()

    preds = infer_model(model, pyg_graph, num_test, **kwargs)
    
    if pl == "batchbald-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, n_clusters, pyg_graph.unlbl_mask, **kwargs)
    elif pl == "batchbald-leiden":
        cluster_labels, _ = leiden_graph(pyg_graph, n_clusters, **kwargs)
    elif pl == "batchbald-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, n_clusters, **kwargs)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]

    selected_indices = np.full([batch], -1, dtype=np.int32)

    preds = preds.detach().squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]

    qtde_clusters_threshold = 0
    for i in range(n_clusters):
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

    l = 0
    start_index = 0
    for i in range(n_clusters):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        num_samples = preds_unlbl_hd.shape[0]
        if num_samples > threshold_cluster:
            qtde = min(qtde_per_cluster[l], num_samples)
            multiplier = min(4, max(1, num_samples // (qtde * 2)))

            if preds_unlbl_hd.dim() == 3 and num_samples >= qtde * 2 * multiplier:
                candidate_batch = batchbald_inverted.get_batchbald_batch_inverted(
                    preds_unlbl_hd, qtde * multiplier, num_samples, dtype=torch.double, device=device
                )
    
            elif preds_unlbl_hd.dim() == 3 and qtde < num_samples < qtde * 2 * multiplier:
                candidate_batch = batchbald.get_bald_batch(
                    preds_unlbl_hd, qtde * multiplier, dtype=torch.double, device=device
                )

            if preds_unlbl_hd.dim() == 3 and num_samples > qtde:
           
                j = 1
                k = 0
                while True:
                    candidate = candidate_batch.indices[-1*j]
                    preds_candidate = preds_unlbl_hd[candidate].argmax(dim=1)
                    if preds_candidate.unique().shape[0] == 1:
                        selected_indices[start_index + k] = unlbl_ix[cluster_indices[candidate]]
                        k += 1
                    j += 1

                    if k == qtde or j > len(candidate_batch.indices):
                        start_index += k
                        break
            l += 1

    return selected_indices


def batchbald_pl(model, pyg_graph, batch, device, bald_iter=10, **kwargs):
    model.eval()

    al_batch = batch
    
    preds_mt= model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    preds_unlbl = preds_mt[pyg_graph.unlbl_mask]
    original_index = torch.argwhere(pyg_graph.unlbl_mask == True)

    candidate_batch = batchbald_inverted.get_batchbald_batch_inverted(
        preds_unlbl, al_batch, al_batch, dtype=torch.double, device=device
    )

    selected_indices = original_index[candidate_batch.indices]

    return selected_indices.cpu()


def bald_pl(model, pyg_graph, batch, device, bald_iter=10, **kwargs):
    model.eval()

    al_batch = batch
    
    preds_mt= model(pyg_graph.x, pyg_graph.edge_index, bald_iter)

    preds_mt = preds_mt.detach()

    preds_unlbl = preds_mt[pyg_graph.unlbl_mask]
    original_index = torch.argwhere(pyg_graph.unlbl_mask == True)

    candidate_batch = batchbald.get_bald_batch(
        preds_unlbl, preds_unlbl.shape[0], dtype=torch.double, device=device
    )

    selected_indices = original_index[candidate_batch.indices[::-1][:al_batch]]

    return selected_indices.cpu()


def pl_update(model, pyg_graph_train, pyg_graph_dev, **kwargs):
    pl = kwargs.get("pl")
    lbl_train_frac = kwargs.get("lbl_train_frac")
    pl_batch = kwargs.get("pl_batch")
    set_id = kwargs.get("set_id")

    if  pl == "bald-kmeans" or pl == "bald-leiden" or pl == "bald-kmeans-aug":
        selected_indices = bald_cluster_pl(model, pyg_graph_train, pl_batch, **kwargs)
        selected_indices = selected_indices.squeeze()
    if  pl == "batchbald-kmeans" or pl == "batchbald-leiden" or pl == "batchbald-kmeans-aug":
        selected_indices = batchbald_cluster_pl(model, pyg_graph_train, pl_batch, **kwargs)
        selected_indices = selected_indices.squeeze()
    if  pl == "unc-kmeans" or pl == "unc-leiden" or pl == "unc-kmeans-aug":
        selected_indices = unc_cluster_pl(model, pyg_graph_train, pl_batch, **kwargs)
        selected_indices = selected_indices.squeeze()
    if  pl == "batchbald":
        selected_indices = batchbald_pl(model, pyg_graph_train, pl_batch, **kwargs)
        selected_indices = selected_indices.squeeze()
    if  pl == "bald":
        selected_indices = bald_pl(model, pyg_graph_train, pl_batch, **kwargs)
        selected_indices = selected_indices.squeeze()

    selected_indices = selected_indices[np.argwhere(selected_indices != -1)].squeeze()

    degree_report = dict()
    degree_report["exp"] = kwargs.get("exp_id")
    degree_report["mode"] = "pseudo-labeling"
    degree_report["labeled_size"] = kwargs.get("lbl_num")
    degree_report["set_id"] = kwargs.get("set_id")
    degree_report["run_id"] = kwargs.get("run_id")

    deg = degree(pyg_graph_train.edge_index[0], num_nodes=pyg_graph_train.num_nodes)
    degree_report["mean_degree"] = torch.mean(deg[selected_indices]).cpu().numpy()
    # degree_report["mean_bw"] = torch.mean(pyg_graph_train.betweenness[selected_indices]).cpu().numpy()

    csv_file = DEGREE_CSV.format(kwargs.get("event"), kwargs.get("exp_group"))

    # Save degree_report to CSV file
    save_to_csv(degree_report, csv_file)

    if selected_indices.shape[0] > 0:
        pyg_graph_train.labeled_mask[selected_indices] = True
        pyg_graph_train.unlbl_mask[selected_indices] = False
        pyg_graph_dev.labeled_mask[selected_indices] = True
        pyg_graph_dev.unlbl_mask[selected_indices] = False
    
        pseudo_train, pseudo_val = train_test_split(selected_indices, train_size=lbl_train_frac, random_state=set_id)
        pyg_graph_train.pseudo_train_mask[pseudo_train] = True
        pyg_graph_train.pseudo_val_mask[pseudo_val] = True
        pyg_graph_dev.pseudo_train_mask[pseudo_train] = True
        pyg_graph_dev.pseudo_val_mask[pseudo_val] = True

    return pyg_graph_train, pyg_graph_dev


def unc_cluster_pl(model, pyg_graph, batch, **kwargs):
    num_test = 1
    pl = kwargs.get("pl")
    threshold_cluster = kwargs.get("threshold_cluster")

    model.eval()

    preds = infer_model(model, pyg_graph, num_test, **kwargs)
    
    if pl == "unc-kmeans":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch, pyg_graph.unlbl_mask, **kwargs)
    elif pl == "unc-leiden":
        cluster_labels, _ = leiden_graph(pyg_graph, batch, **kwargs)
    elif pl == "unc-kmeans-aug":
        cluster_labels, _ = kmeans_graph(pyg_graph, batch, **kwargs)
        cluster_labels = cluster_labels[pyg_graph.unlbl_mask.cpu().numpy()]

    selected_indices = np.full([batch], -1, dtype=np.int32)

    preds = preds.squeeze()
    unlbl_ix = torch.argwhere(pyg_graph.unlbl_mask == True).squeeze().cpu().numpy()
    preds = preds[unlbl_ix]

    for i in range(batch):
        cluster_indices = np.argwhere(cluster_labels == i).squeeze()
        preds_unlbl_hd = preds[cluster_indices]
        if preds_unlbl_hd.dim() == 2:
            if preds_unlbl_hd.shape[0] >= threshold_cluster:
                preds_unlbl_hd = torch.max(preds_unlbl_hd, dim=1).values
                candidate = torch.argmax(preds_unlbl_hd)
                selected_indices[i] = unlbl_ix[cluster_indices[candidate]]

    return selected_indices