import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree
import torch
import igraph as ig
import leidenalg as la

def data_split(pyg_graph, **kwargs):
    isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    lbl_train_frac = kwargs.get("lbl_train_frac")
    aug_unlbl_set = kwargs.get("aug_unlbl_set")

    event_ix = torch.argwhere((pyg_graph.y == 0) | (pyg_graph.y == 1) ).squeeze()
    ft_full = pyg_graph.x.cpu().detach()
    ft = ft_full[event_ix]

    qtde_items = ft.shape[0]
        
    pseudo_train_size = int(labeled_size * lbl_train_frac)
    pseudo_val_size = labeled_size - pseudo_train_size

    if isel == "random":
        labeled_ix, unlbl_ix = train_test_split(event_ix, train_size=labeled_size, random_state=set_id)
        pseudo_train, pseudo_val = train_test_split(labeled_ix, train_size=lbl_train_frac, random_state=set_id)
    
        return labeled_ix, unlbl_ix, pseudo_train, pseudo_val

    elif isel == "kmeans":
        kmeans = KMeans(n_clusters=pseudo_train_size, random_state=set_id)

        kmeans.fit(ft_full)
        cluster_labels = kmeans.labels_
    
        pseudo_train = torch.full([pseudo_train_size], -1, dtype=torch.int)

        for i in range(pseudo_train_size):
            # Find the indices of samples closest to cluster i
            indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((pyg_graph.y == 0) | (pyg_graph.y == 1)) ).squeeze(dim=1)

            # Calculate the distance between each sample in the cluster and the centroid of cluster i
            distances = torch.norm(ft[indices_i] - kmeans.cluster_centers_[i], dim=1)

            # Find the index of the sample with the minimum distance
            min_index = indices_i[torch.argmin(distances)]
            pseudo_train[i] = min_index
        
    if isel == "kmeans-degree":
        kmeans = KMeans(n_clusters=pseudo_train_size, random_state=set_id)

        kmeans.fit(ft_full)
        cluster_labels = kmeans.labels_
    
        pseudo_train = torch.zeros([pseudo_train_size], dtype=torch.int)

        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        for i in range(pseudo_train_size):
            # Find the indices of samples closest to cluster i
            indices_i = torch.nonzero((torch.tensor(cluster_labels) == i) & ((pyg_graph.y == 0) | (pyg_graph.y == 1)) ).squeeze(dim=1)

            degrees_i = deg[indices_i]
            # Find the index of the sample with the minimum distance
            min_index = indices_i[torch.argmax(degrees_i)]
            pseudo_train[i] = min_index
        
    if isel == "degree":
        # Calculate the degree of each node
        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        deg = deg[event_ix]

        # Sort nodes based on their degree
        sorted_nodes = torch.argsort(deg, descending=True)

        # Select the top k nodes
        pseudo_train = event_ix[sorted_nodes[:pseudo_train_size]]

    if isel == "leiden":
        pseudo_train = torch.zeros([pseudo_train_size], dtype=torch.int)

        edge_index = pyg_graph.edge_index
        num_nodes = pyg_graph.num_nodes
        
        # Create igraph.Graph
        G = ig.Graph(n=num_nodes)
        G.add_edges(edge_index.T.tolist())  # Convert edge_index to list of edges

        partition = la.find_partition(G, la.ModularityVertexPartition);

        # Calculate the degree of each node
        deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

        qtde_partitions = len(partition)

        if pseudo_train_size <= qtde_partitions:
            for i in range(pseudo_train_size):
                points_partition = partition[i]
                points_partition_event = torch.isin(points_partition, event_ix)
                
                max_deg_ix = torch.argmax(deg[points_partition_event])
    
                pseudo_train[i] = partition[i][max_deg_ix]
        else:
            qtde_per_group = pseudo_train_size // qtde_partitions
            one_more = pseudo_train_size % qtde_partitions
            
            i_begin = 0
            for i in range(qtde_partitions):
                points_partition = partition[i]
                points_partition_event = torch.isin(points_partition, event_ix)

                if i < one_more:
                    max_deg_ix = torch.topk(deg[points_partition_event], k=qtde_per_group + 1)
                    i_end = i_begin + qtde_per_group + 1
                else:
                    max_deg_ix = torch.topk(deg[points_partition_event], k=qtde_per_group)
                    i_end = i_begin + qtde_per_group
            
                pseudo_train[i_begin:i_end] = torch.Tensor(partition[i])[max_deg_ix.indices]
                i_begin = i_end

    # random val
    pseudo_val = torch.Tensor(np.random.choice([i for i in event_ix if i not in pseudo_train], pseudo_val_size))
    pseudo_val = pseudo_val.to(torch.int)

    labeled_ix = torch.concat([pseudo_train, pseudo_val])
    unlbl_ix = torch.Tensor([i for i in event_ix if i not in labeled_ix])
    unlbl_ix = unlbl_ix.to(torch.int)

    return labeled_ix, unlbl_ix, pseudo_train, pseudo_val
