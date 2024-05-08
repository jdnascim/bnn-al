import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree
import torch

def data_split(pyg_graph, **kwargs):
    isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    lbl_train_frac = kwargs.get("lbl_train_frac")
    
    ft = pyg_graph.x.cpu().detach()

    qtde_items = ft.shape[0]

    if isel == "random":
        labeled_ix, unlbl_ix = train_test_split(np.arange(qtde_items), train_size=labeled_size, random_state=set_id)
        pseudo_train, pseudo_val = train_test_split(labeled_ix, train_size=lbl_train_frac, random_state=set_id)
    if isel == "kmeans":
        pseudo_train_size = int(labeled_size * lbl_train_frac)
        pseudo_val_size = labeled_size - pseudo_train_size

        kmeans = KMeans(n_clusters=pseudo_train_size, random_state=set_id)

        kmeans.fit(ft)
        cluster_labels = kmeans.labels_
    
        pseudo_train = torch.zeros([pseudo_train_size], dtype=torch.int)

        for i in range(pseudo_train_size):
            # Find the indices of samples closest to cluster i
            indices_i = torch.nonzero(torch.tensor(cluster_labels) == i).squeeze(dim=1)
            # Calculate the distance between each sample in the cluster and the centroid of cluster i
            distances = torch.norm(ft[indices_i] - kmeans.cluster_centers_[i], dim=1)
            # Find the index of the sample with the minimum distance
            min_index = indices_i[torch.argmin(distances)]
            pseudo_train[i] = min_index
        
        # random val
        pseudo_val = torch.Tensor(np.random.choice([i for i in range(qtde_items) if i not in pseudo_train], pseudo_val_size))
        pseudo_val = pseudo_val.to(torch.int)

        labeled_ix = torch.concat([pseudo_train, pseudo_val])
        unlbl_ix = torch.Tensor([i for i in range(qtde_items) if i not in labeled_ix])
        unlbl_ix = unlbl_ix.to(torch.int)
    if isel == "kmeans-degree":
        pseudo_train_size = int(labeled_size * lbl_train_frac)
        pseudo_val_size = labeled_size - pseudo_train_size

        kmeans = KMeans(n_clusters=pseudo_train_size, random_state=set_id)

        kmeans.fit(ft)
        cluster_labels = kmeans.labels_
    
        pseudo_train = torch.zeros([pseudo_train_size], dtype=torch.int)

        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        for i in range(pseudo_train_size):
            # Find the indices of samples closest to cluster i
            indices_i = torch.nonzero(torch.tensor(cluster_labels) == i).squeeze(dim=1)
            # Calculate the distance between each sample in the cluster and the centroid of cluster i
            distances = torch.norm(ft[indices_i] - kmeans.cluster_centers_[i], dim=1)

            degrees_i = deg[indices_i]
            # Find the index of the sample with the minimum distance
            min_index = indices_i[torch.argmax(degrees_i)]
            pseudo_train[i] = min_index
        
        # random val
        pseudo_val = torch.Tensor(np.random.choice([i for i in range(qtde_items) if i not in pseudo_train], pseudo_val_size))
        pseudo_val = pseudo_val.to(torch.int)

        labeled_ix = torch.concat([pseudo_train, pseudo_val])
        unlbl_ix = torch.Tensor([i for i in range(qtde_items) if i not in labeled_ix])
        unlbl_ix = unlbl_ix.to(torch.int)
    if isel == "degree":
        pseudo_train_size = int(labeled_size * lbl_train_frac)
        pseudo_val_size = labeled_size - pseudo_train_size

        # Calculate the degree of each node
        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        # Sort nodes based on their degree
        sorted_nodes = torch.argsort(deg, descending=True)

        # Select the top k nodes
        pseudo_train = sorted_nodes[:pseudo_train_size]

        # random val
        pseudo_val = torch.Tensor(np.random.choice([i for i in range(qtde_items) if i not in pseudo_train], pseudo_val_size))
        pseudo_val = pseudo_val.to(torch.int)

        labeled_ix = torch.concat([pseudo_train, pseudo_val])
        unlbl_ix = torch.Tensor([i for i in range(qtde_items) if i not in labeled_ix])
        unlbl_ix = unlbl_ix.to(torch.int)

    return labeled_ix, unlbl_ix, pseudo_train, pseudo_val
