import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree, to_networkx
import torch
import igraph as ig
import leidenalg as la
from src.al.utils import kmeans_graph, leiden_graph
import networkx as nx

def data_split(pyg_graph, **kwargs):
    isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    lbl_train_frac = kwargs.get("lbl_train_frac")
    hetero = kwargs.get("hetero")
    threshold_cluster = kwargs.get("threshold_cluster")
    random_pseudo_val = kwargs.get("random_pseudo_val")

    assert hetero is False or isel == "random", "isel not implemented yet for heterogenous graphs"

    if hetero is True:
        lbl = pyg_graph["joint"].y
        ft_full = pyg_graph["joint"].x.cpu().detach()
    else:
        lbl = pyg_graph.y
        ft_full = pyg_graph.x.cpu().detach()

    event_ix = torch.argwhere((lbl == 0) | (lbl == 1) ).squeeze()

    ft = ft_full

    qtde_items = ft.shape[0]
        
    if random_pseudo_val:
        pseudo_train_size = int(labeled_size * lbl_train_frac)
        pseudo_val_size = labeled_size - pseudo_train_size
    else:
        pseudo_train_size = labeled_size

    if isel == "random":
        labeled_ix, unlbl_ix = train_test_split(event_ix, train_size=labeled_size, random_state=set_id)
        pseudo_train, pseudo_val = train_test_split(labeled_ix, train_size=lbl_train_frac, random_state=set_id)
    
        return labeled_ix, unlbl_ix, pseudo_train, pseudo_val

    elif isel == "kmeans" or isel == "kmeans-th":
        if isel == "kmeans":
            threshold_cluster = 0

        cluster_labels, cluster_centers = kmeans_graph(pyg_graph, pseudo_train_size, **kwargs)

        qtde_clusters_threshold = 0
        for i in range(pseudo_train_size):
            cluster_indices = np.argwhere(cluster_labels == i).squeeze()
            indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((lbl == 0) | (lbl == 1)) ).squeeze(dim=1)
            num_samples = indices_i.shape[0]

            if num_samples > threshold_cluster:
                qtde_clusters_threshold += 1
        
        if qtde_clusters_threshold == 0:
            qtde_clusters_threshold = pseudo_train_size
            threshold_cluster = 0
            
        qtde_per_cluster = np.full(qtde_clusters_threshold, pseudo_train_size // qtde_clusters_threshold, dtype=np.int32)
        qtde_per_cluster[:pseudo_train_size % qtde_clusters_threshold] += 1
    
        pseudo_train = torch.full([pseudo_train_size], -1, dtype=torch.int)

        if isel == "kmeans":
            for i in range(pseudo_train_size):
                # Find the indices of samples closest to cluster i
                indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((lbl == 0) | (lbl == 1)) ).squeeze(dim=1)
    # 
                if indices_i.shape[0] > 0:
    # 
                    # Calculate the distance between each sample in the cluster and the centroid of cluster i
                    distances = torch.norm(ft[indices_i] - cluster_centers[i], dim=1)
        # 
                    # Find the index of the sample with the minimum distance
                    min_index = indices_i[torch.argmin(distances)]
                    pseudo_train[i] = min_index
        elif isel == "kmeans-th":
            j = 0
            start_ix = 0
            for i in range(pseudo_train_size):
                cluster_indices = np.argwhere(cluster_labels == i).squeeze()
    
                indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((lbl == 0) | (lbl == 1)) ).squeeze(dim=1)
                num_samples = indices_i.shape[0]
    
                if num_samples > threshold_cluster and indices_i.shape[0] > 0:
                    qtde = min(qtde_per_cluster[j], num_samples)
                    end_ix = start_ix + qtde
    
                    distances = torch.norm(ft[indices_i] - cluster_centers[i], dim=1)
                    _, indices = torch.topk(distances, qtde, largest=False)
    
                    pseudo_train[start_ix:end_ix] = indices_i[indices]
                    start_ix = end_ix
                    
                    j += 1
        
    elif isel == "kmeans-degree":
        cluster_labels, _ = kmeans_graph(pyg_graph, pseudo_train_size, **kwargs)
    
        pseudo_train = torch.full([pseudo_train_size], -1, dtype=torch.int)

        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        for i in range(pseudo_train_size):
            # Find the indices of samples closest to cluster i
            indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((lbl == 0) | (lbl == 1)) ).squeeze(dim=1)

            if indices_i.shape[0] > 0:
                degrees_i = deg[indices_i]
                # Find the index of the sample with the minimum distance
                min_index = indices_i[torch.argmax(degrees_i)]
                pseudo_train[i] = min_index
        
    elif isel == "degree":
        # Calculate the degree of each node
        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        deg = deg[event_ix]

        # Sort nodes based on their degree
        sorted_nodes = torch.argsort(deg, descending=True)

        # Select the top k nodes
        pseudo_train = event_ix[sorted_nodes[:pseudo_train_size]]

    elif isel == "leiden" or isel == "leiden-ic" or isel == "leiden-bw":
        pseudo_train = torch.full([pseudo_train_size], -1, dtype=torch.int32)

        cluster_labels = leiden_graph(pyg_graph)

        if isel == "leiden":
        # Calculate the degree of each node
            deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)
        elif isel == "leiden-ic":
            deg = torch.zeros(pyg_graph.num_nodes, dtype=torch.int32)
            for edge in pyg_graph.edge_index.T:
                src, dst = edge
                if cluster_labels[src] == cluster_labels[dst]:
                    deg[src] += 1
                    deg[dst] += 1
        elif isel == "leiden-bw":
            # Convert the PyTorch Geometric graph to a NetworkX graph
            G = to_networkx(pyg_graph, to_undirected=True)
            betweenness_centrality = {}
            
            # Calculate betweenness centrality within each cluster
            unique_clusters = np.unique(cluster_labels)
            for cluster in unique_clusters:
                cluster_nodes = (cluster_labels == cluster).nonzero()[0].tolist()
                subgraph = G.subgraph(cluster_nodes)
                bw_centrality = nx.betweenness_centrality(subgraph)
                
                # Update the betweenness centrality dictionary without a for loop
                betweenness_centrality.update(bw_centrality)
            
            # Convert betweenness centrality to a tensor
            deg = torch.tensor([betweenness_centrality.get(i, 0) for i in range(pyg_graph.num_nodes)], dtype=torch.float)

        qtde_partitions = np.unique(cluster_labels).shape[0]

        if pseudo_train_size <= qtde_partitions:
            for i in range(pseudo_train_size):
                points_partition = torch.Tensor(np.argwhere(cluster_labels == i)).squeeze().int()
                points_partition_ix = torch.isin(points_partition, event_ix)
                points_partition_event = points_partition[points_partition_ix].int()
                
                if points_partition_event.shape[0] > 0:
                    max_deg_ix = torch.argmax(deg[points_partition_event])
    
                    pseudo_train[i] = points_partition[max_deg_ix]
        else:
            qtde_per_group = pseudo_train_size // qtde_partitions
            one_more = pseudo_train_size % qtde_partitions
            
            i_begin = 0
            for i in range(qtde_partitions):
                points_partition = torch.Tensor(np.argwhere(cluster_labels == i)).squeeze().int()
                points_partition_ix = torch.isin(points_partition, event_ix)
                points_partition_event = points_partition[points_partition_ix].int()

                if i < one_more:
                    i_end = i_begin + qtde_per_group + 1
                    if points_partition_event.shape[0] > 0:
                        k = min(qtde_per_group + 1, points_partition_event.shape[0])
                        max_deg_ix = torch.topk(deg[points_partition_event], k=k)
                        pseudo_train[i_begin:i_begin+k] = points_partition[max_deg_ix.indices].int()
                else:
                    i_end = i_begin + qtde_per_group
                    if points_partition_event.shape[0] > 0:
                        k = min(qtde_per_group, points_partition_event.shape[0])
                        max_deg_ix = torch.topk(deg[points_partition_event], k=k)
                        pseudo_train[i_begin:i_begin+k] = points_partition[max_deg_ix.indices].int()

                i_begin = i_end

    # add a random item in case one cluster does not have event data enough. 
    negative_indices = (pseudo_train == -1).nonzero(as_tuple=False)
    qtde_negative_indices = len(negative_indices)

    if random_pseudo_val:
        pseudo_train = pseudo_train.int()
        pseudo_train[negative_indices.squeeze()] = torch.tensor(np.random.choice(event_ix[~torch.isin(event_ix, pseudo_train)], size=qtde_negative_indices, replace=False)).squeeze().int()
    
        # random val
        pseudo_val = torch.Tensor(np.random.choice([i for i in event_ix if i not in pseudo_train], pseudo_val_size))
        pseudo_val = pseudo_val.to(torch.int)
    else:
        pseudo_train, pseudo_val = train_test_split(pseudo_train, train_size=lbl_train_frac, random_state=set_id)

    labeled_ix = torch.concat([pseudo_train, pseudo_val])
    unlbl_ix = torch.Tensor([i for i in event_ix if i not in labeled_ix])
    unlbl_ix = unlbl_ix.to(torch.int)

    return labeled_ix, unlbl_ix, pseudo_train, pseudo_val
