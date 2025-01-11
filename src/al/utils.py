from sklearn.cluster import KMeans
import igraph as ig
import leidenalg as la
import numpy as np
import torch

from src.utils.utils import tensor_to_numpy


def kmeans_graph(pyg_graph, n_clusters, mask=None, **kwargs):
    hetero = kwargs.get("hetero")
    set_id = kwargs.get("set_id")

    kmeans = KMeans(n_clusters=n_clusters, random_state=set_id)

    if hetero is True:
        feature_matrix = tensor_to_numpy(pyg_graph["joint"].x)
    else:
        feature_matrix = tensor_to_numpy(pyg_graph.x)

    if mask is not None:
        mask = tensor_to_numpy(mask)
        feature_matrix = feature_matrix[mask]

    kmeans.fit(feature_matrix)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


def leiden_graph(pyg_graph, **kwargs):
    hetero = kwargs.get("hetero")

    if hetero is True:
        edge_index = pyg_graph["joint"].edge_index
        num_nodes = pyg_graph["joint"].num_nodes
    else:
        edge_index = pyg_graph.edge_index
        num_nodes = pyg_graph.num_nodes
    
    # Create igraph.Graph
    G = ig.Graph(n=num_nodes)
    G.add_edges(edge_index.T.tolist())  # Convert edge_index to list of edges

    partition = la.find_partition(G, la.ModularityVertexPartition);

    cluster_labels = np.zeros([num_nodes], dtype=np.int32)

    for i, p in enumerate(partition):
        cluster_labels[p] = i
    
    return cluster_labels