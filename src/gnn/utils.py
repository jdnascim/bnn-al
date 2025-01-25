import copy
import json
import os
import pickle
import statistics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data, HeteroData
import tqdm
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from torch_geometric.utils import degree
import igraph as ig
import networkx as nx

from src.al.isel import data_split
from src.feature_extraction.feature_extraction import clip_features, mpnet_features, maxvit_features
from src.utils.utils import check_and_convert_to_tensor, custom_serializer, get_f1, get_normalized_acc, process_dataframe, remove_duplicates
from src.utils.constants import AL_SPLIT_SET, AUG_GRAPH_CACHE, DEV_SET, EVENT_AUG_DIFF, EVENTS, GRAPH_CACHE, TRAIN_SET, EVENT_AUG_PAIRS
from src.utils.reduction import autoencoder_reduction

def infer_model(model, pyg_graph, k, **kwargs):
    hetero = kwargs.get("hetero")
    sml = kwargs.get("single_label_loss")

    if hetero is True:
        preds = model(pyg_graph.x_dict, pyg_graph.edge_index_dict, k)
    else:
        preds = model(pyg_graph.x, pyg_graph.edge_index, k)
    
    if bool(sml) is True:
        preds = preds[-1]
    
    return preds


def concat_ref_matrix(A, B):
    # Get their shapes
    shape_A = A.shape
    shape_B = B.shape
    
    # Sum the shapes element-wise
    summed_shape = (shape_A[0] + shape_B[0], shape_A[1] + shape_B[1])

    concat = torch.zeros(summed_shape)

    concat[:A.shape[0], :A.shape[1]] = A
    concat[A.shape[0]:, A.shape[1]:] = B

    return concat


def highest_degree_unlbl_nodes(pyg_graph, k):

    assert 1 <= pyg_graph.x[pyg_graph.unlbl_mask].shape[0], "k smaller than 1"
    assert k <= pyg_graph.x[pyg_graph.unlbl_mask].shape[0], "k bigger than unlabeled set size"

    # Calculate the degree of each node
    deg = degree(pyg_graph.edge_index[0], num_nodes=pyg_graph.num_nodes)

    # Consider only the unlabeled_set
    deg = deg[pyg_graph.unlbl_mask]
    original_index = torch.argwhere(pyg_graph.unlbl_mask == True)

    # Sort nodes based on their degree
    sorted_nodes = torch.argsort(deg, descending=True)

    # Select the top k nodes
    top_k_nodes = original_index[sorted_nodes[:k]].squeeze()

    return top_k_nodes

def adj_matrix_to_edge_list(adj_matrix):
    # Find the indices of the non-zero elements in the upper triangle of the matrix
    rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
    
    # Stack them to get the edges in the form [2, n_edges]
    edges = torch.stack((rows, cols))

    # Order based on joint index
    sorted_indices = torch.argsort(edges[1, :])
    sorted_edge_list = edges[:, sorted_indices]
    
    return sorted_edge_list


def graph_edges(emb, n_neighbors, mode="knn", max_sim_iter=100):

    if torch.is_tensor(emb) is True:
        device = emb.device
        emb = emb.detach().cpu().numpy()
    
    simm = cosine_similarity(emb)
    emb = torch.Tensor(emb).to(device)

    simm[np.arange(simm.shape[0]),np.arange(simm.shape[0])] = 0

    if mode == "knn":
        edges = set()        
        for i, vec in enumerate(simm):
            partit = np.argpartition(vec, -1*n_neighbors)
            for j in range(n_neighbors):
                edges.add((i, partit[-1 * j]))
                edges.add((partit[-1 * j], i))
            
        edges = torch.Tensor(list(edges)).t().type(torch.int64)
    elif mode == "sim":
        num_nodes = simm.shape[0]
        # Initialize graph with empty edges
        
        max_threshold = np.amax(simm, axis=1).max()

        # threshold = (max_threshold + min_threshold) / 2
        threshold = max_threshold
        scaler = 0.1

        # Set flag to keep iterating until mean degree reaches k
        ix = 0
        while True:
            graph = ig.Graph(n=num_nodes)
            # Threshold the similarity matrix
            adjacency_matrix = simm > threshold
            
            # Convert adjacency matrix to list of edges
            edges_ix = np.transpose(np.where(adjacency_matrix))
            
            # Update graph with edges
            graph.add_edges(edges_ix)
            
            # Calculate mean degree
            mean_degree = np.mean(graph.degree())
            
            #Check if mean degree is close to k
            if abs(mean_degree - n_neighbors) < 0.1 or ix >= max_sim_iter:  # Adjust tolerance as needed
                break
            elif mean_degree < n_neighbors:  # If mean degree is less than k, decrease threshold
                threshold -= scaler
                # threshold = (min_threshold + threshold) / 2
            else:  # If mean degree is greater than k, increase threshold
                threshold += scaler
                scaler *= 0.1
                threshold -= scaler
                # threshold = (max_threshold + threshold) / 2

            ix += 1
        
        # Convert igraph edges to torch_geometric edges
        edges = np.array(graph.get_edgelist()).T
        edges = torch.tensor(edges, dtype=torch.long)

    elif mode == "sim-connected":
        num_nodes = simm.shape[0]
        # Initialize graph with empty edges
        
        max_threshold = np.amax(simm, axis=1).max()
        min_threshold = np.sort(simm, axis=1)[:, ::-1][:, 1].min()

        threshold = min_threshold
        scaler = 0.1

        # Set flag to keep iterating until mean degree reaches k
        ix = 0
        while True:
            graph = ig.Graph(n=num_nodes)
            # Threshold the similarity matrix
            adjacency_matrix = simm > threshold
            
            # Convert adjacency matrix to list of edges
            edges_ix = np.transpose(np.where(adjacency_matrix))
            
            # Update graph with edges
            graph.add_edges(edges_ix)
            
            # Calculate mean degree
            mean_degree = np.mean(graph.degree())
            
            #Check if mean degree is close to k
            if abs(mean_degree - n_neighbors) < 0.1 or ix >= max_sim_iter:  # Adjust tolerance as needed
                break
            elif threshold == min_threshold and mean_degree > n_neighbors:
                break
            elif mean_degree < n_neighbors:  # If mean degree is less than k, decrease threshold
                threshold = (min_threshold + threshold) / 2
            else:  # If mean degree is greater than k, increase threshold
                threshold = (max_threshold + threshold) / 2

            ix += 1

        # Convert igraph edges to torch_geometric edges
        edges = np.array(graph.get_edgelist()).T
        edges = torch.tensor(edges, dtype=torch.long)
        
    return edges
    

def generate_graph_hetero(**kwargs):
    event = kwargs.get("event")
    reduction = kwargs.get("reduction")
    al_isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    cache = kwargs.get("use_cache")
    n_neigh_train = kwargs.get("n_neigh_train")
    n_neigh_full = kwargs.get("n_neigh_full")
    graph_mode = kwargs.get("graph_mode")
    imageft = kwargs.get("imageft")
    textft = kwargs.get("textft")
    aug_unlbl_set = kwargs.get("aug_unlbl_set")
    device = kwargs.get("device")

    if aug_unlbl_set == "one":
        aug_event = EVENT_AUG_PAIRS[event]
        filepath = AUG_GRAPH_CACHE.format(event, aug_event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    if aug_unlbl_set == "all":
        filepath = AUG_GRAPH_CACHE.format(event, "all", imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    if aug_unlbl_set == "all_differ":
        filepath = AUG_GRAPH_CACHE.format(event, "all_differ", imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    else:
        filepath = GRAPH_CACHE.format(event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)

    filepath = filepath.replace("/graph", '/heterograph')

    if textft == "mpnet":
        [df_text_train, df_text_dev, _] = mpnet_features(**kwargs)
    elif textft == "clip":
        [df_text_train, df_text_dev, _] = clip_features(mode="text", **kwargs)

    if imageft == "maxvit":
        [df_image_train, df_image_dev, _] = maxvit_features(**kwargs)
    elif textft == "clip":
        [df_image_train, df_image_dev, _] = clip_features(mode="image", **kwargs)


    [df_text_train_joint, df_text_dev_joint, _] = clip_features(mode="text", **kwargs)
    [df_image_train_joint, df_image_dev_joint, _] = clip_features(mode="image", **kwargs)

    data_train = pd.read_json(TRAIN_SET.format(event), lines=True)
    ft_train_images, ft_train_text, annot_train, annot_train_images, annot_train_text = process_dataframe(data_train,
                                                                                                        df_image_train,
                                                                                                        df_text_train)

    ft_train_images_joint, ft_train_text_joint, _, _, _, = process_dataframe(data_train,
                                                                            df_image_train_joint,
                                                                            df_text_train_joint)

    # dev
    data_dev = pd.read_json(DEV_SET.format(event), lines=True)
    ft_dev_images, ft_dev_text, annot_dev, annot_dev_images, annot_dev_text = process_dataframe(data_dev, df_image_dev, df_text_dev)
    ft_dev_images_joint, ft_dev_text_joint, _, _, _ = process_dataframe(data_dev, df_image_dev_joint, df_text_dev_joint)

    if aug_unlbl_set is not None:
        annot_train = np.column_stack([annot_train, np.zeros(annot_train.shape[0])])
        annot_train_images = np.column_stack([annot_train_images, np.zeros(annot_train.shape[0])])
        annot_train_text = np.column_stack([annot_train_text, np.zeros(annot_train.shape[0])])

        if aug_unlbl_set == "one":
            aug_event = [EVENT_AUG_PAIRS[event]]
        elif aug_unlbl_set == "all_differ":
            aug_event = EVENT_AUG_DIFF[event]
        elif aug_unlbl_set == "all":
            aug_event = [e for e in EVENTS if e != event]

        df_image_aug = None
        df_text_aug = None
            
        for ae in aug_event:
            if textft == "mpnet":
                [df_text_aug, _, _] = mpnet_features(event_features=ae, **kwargs)
            elif textft == "clip":
                [df_text_aug, _, _] = clip_features(event_features=ae, mode="text", **kwargs)
        
            if imageft == "maxvit":
                [df_image_aug, _, _] = maxvit_features(event_features=ae, **kwargs)
            elif textft == "clip":
                [df_image_aug, _, _] = clip_features(event_features=ae, mode="image", **kwargs)

            [df_text_aug_joint, _, _] = clip_features(event_features=ae, mode="text", **kwargs)
            [df_image_aug_joint, _, _] = clip_features(event_features=ae, mode="image", **kwargs)
            
            df_image_aug["labels"] = 2
            df_text_aug["labels"] = 2
            df_image_aug_joint["labels"] = 2
            df_text_aug_joint["labels"] = 2

            data_aug = pd.read_json(TRAIN_SET.format(ae), lines=True)
            ft_aug_images, ft_aug_text, annot_aug, annot_aug_images, annot_aug_text = process_dataframe(data_aug, df_image_aug, df_text_aug)
            ft_aug_images_joint, ft_aug_text_joint, _, _, _ = process_dataframe(data_aug, df_image_aug_joint, df_text_aug_joint)

            annot_aug = np.zeros_like(annot_aug)
            annot_aug = np.column_stack([annot_aug, np.ones(annot_aug.shape[0])])
            annot_aug_images = np.zeros_like(annot_aug_images)
            annot_aug_images = np.column_stack([annot_aug_images, np.ones(annot_aug_images.shape[0])])
            annot_aug_text = np.zeros_like(annot_aug_text)
            annot_aug_text = np.column_stack([annot_aug_text, np.ones(annot_aug_text.shape[0])])

            ft_train_images = torch.concat([ft_train_images, ft_aug_images])
            ft_train_text = torch.concat([ft_train_text, ft_aug_text])
            ft_train_images_joint = torch.concat([ft_train_images_joint, ft_aug_images_joint])
            ft_train_text_joint = torch.concat([ft_train_text_joint, ft_aug_text_joint])
            annot_train = np.concatenate([annot_train, annot_aug])
            annot_train_images = np.concatenate([annot_train_images, annot_aug_images])
            annot_train_text = np.concatenate([annot_train_text, annot_aug_text])
    
    print(AL_SPLIT_SET.format(event, al_isel, "labeled", labeled_size, set_id))

    ft_train_joint = torch.concat([ft_train_images_joint, ft_train_text_joint], axis=1)
    ft_dev_joint = torch.concat([ft_dev_images_joint, ft_dev_text_joint], axis=1)
    
    ft_train_joint = check_and_convert_to_tensor(ft_train_joint)
    ft_train_images = check_and_convert_to_tensor(ft_train_images)
    ft_train_text = check_and_convert_to_tensor(ft_train_text)
    annot_train = check_and_convert_to_tensor(annot_train)
    annot_dev = check_and_convert_to_tensor(annot_dev)
    annot_train_images = check_and_convert_to_tensor(annot_train_images)
    annot_train_text = check_and_convert_to_tensor(annot_train_text)
    annot_dev_images = check_and_convert_to_tensor(annot_dev_images)
    annot_dev_text = check_and_convert_to_tensor(annot_dev_text)
    
    annot_train = torch.argmax(annot_train, dim=1)
    annot_dev = torch.argmax(annot_dev, dim=1)
    annot_train_images = torch.argmax(annot_train_images, dim=1)
    annot_train_text = torch.argmax(annot_train_text, dim=1)
    annot_dev_images = torch.argmax(annot_dev_images, dim=1)
    annot_dev_text = torch.argmax(annot_dev_text, dim=1)

    ft_train_images, ref_edges_images = remove_duplicates(ft_train_images)
    ft_train_text, ref_edges_text = remove_duplicates(ft_train_text)

    edges_images = graph_edges(ft_train_images, n_neigh_train, graph_mode)
    edges_text = graph_edges(ft_train_text, n_neigh_train, graph_mode)
    edges_joint = graph_edges(ft_train_joint, n_neigh_train, graph_mode)

    pyg_graph_train = HeteroData()

    pyg_graph_train['image'].x = ft_train_images.float()
    pyg_graph_train['text'].x = ft_train_text.float()
    pyg_graph_train['joint'].x = ft_train_joint.float()

    pyg_graph_train["image", "sim", "image"].edge_index = edges_images
    pyg_graph_train["text", "sim", "text"].edge_index = edges_text
    pyg_graph_train["joint", "sim", "joint"].edge_index = edges_joint
    pyg_graph_train["image", "ref", "joint"].edge_index = adj_matrix_to_edge_list(ref_edges_images)
    pyg_graph_train["text", "ref", "joint"].edge_index = adj_matrix_to_edge_list(ref_edges_text)
    pyg_graph_train["image"].y = annot_train_images[torch.argmax(ref_edges_images, dim=1)]
    pyg_graph_train["text"].y = annot_train_text[torch.argmax(ref_edges_text, dim=1)]
    pyg_graph_train["joint"].y = annot_train

    labeled_ix, unlabeled_ix, pseudo_train_ix, pseudo_val_ix = data_split(pyg_graph_train, **kwargs)
    qtde_emb = pyg_graph_train["joint"].x.shape[0]

    labeled_mask = torch.zeros(qtde_emb, dtype=bool)
    labeled_mask[labeled_ix] = 1
    pyg_graph_train.labeled_mask = labeled_mask

    unlabeled_mask = torch.zeros(qtde_emb, dtype=bool)
    unlabeled_mask[unlabeled_ix] = 1
    pyg_graph_train.unlbl_mask = unlabeled_mask

    pseudo_train = torch.zeros(qtde_emb, dtype=bool)
    pseudo_train[pseudo_train_ix] = 1
    pyg_graph_train.pseudo_train_mask = pseudo_train

    pseudo_val = torch.zeros(qtde_emb, dtype=bool)
    pseudo_val[pseudo_val_ix] = 1
    pyg_graph_train.pseudo_val_mask = pseudo_val

    aug_event_mask = torch.ones(qtde_emb, dtype=bool)
    aug_event_mask[((pyg_graph_train["joint"].y == 0) | (pyg_graph_train["joint"].y == 1))] = 0
    pyg_graph_train.aug_event_mask = aug_event_mask

    # dev
    ft_dev_images, ref_edges_images_dev = remove_duplicates(ft_dev_images)
    ft_dev_text, ref_edges_text_dev = remove_duplicates(ft_dev_text)

    ft_dev_images = torch.concat([ft_train_images, ft_dev_images])
    ft_dev_text = torch.concat([ft_train_text, ft_dev_text])
    ft_dev_joint = torch.concat([ft_train_joint, ft_dev_joint])
    annot_dev = torch.concat([annot_train, annot_dev])

    ref_edges_images_full = concat_ref_matrix(ref_edges_images, ref_edges_images_dev)
    ref_edges_text_full = concat_ref_matrix(ref_edges_text, ref_edges_text_dev)

    edges_images = graph_edges(ft_dev_images, n_neigh_full, graph_mode)
    edges_text = graph_edges(ft_dev_text, n_neigh_full, graph_mode)
    edges_joint = graph_edges(ft_dev_joint, n_neigh_full, graph_mode)

    pyg_graph_dev = HeteroData()

    pyg_graph_dev['image'].x = ft_dev_images.float()
    pyg_graph_dev['text'].x = ft_dev_text.float()
    pyg_graph_dev['joint'].x = ft_dev_joint.float()

    pyg_graph_dev["image", "sim", "image"].edge_index = edges_images
    pyg_graph_dev["text", "sim", "text"].edge_index = edges_text
    pyg_graph_dev["joint", "sim", "joint"].edge_index = edges_joint
    pyg_graph_dev["image", "ref", "joint"].edge_index = adj_matrix_to_edge_list(ref_edges_images_full)
    pyg_graph_dev["text", "ref", "joint"].edge_index = adj_matrix_to_edge_list(ref_edges_text_full)
    pyg_graph_dev["image"].y = annot_dev_images
    pyg_graph_dev["text"].y = annot_dev_text
    pyg_graph_dev["joint"].y = annot_dev

    qtde_emb = pyg_graph_dev["joint"].x.shape[0]

    labeled_mask = torch.zeros(qtde_emb, dtype=bool)
    labeled_mask[labeled_ix] = 1
    pyg_graph_dev.labeled_mask = labeled_mask

    unlabeled_mask = torch.zeros(qtde_emb, dtype=bool)
    unlabeled_mask[unlabeled_ix] = 1
    pyg_graph_dev.unlbl_mask = unlabeled_mask

    pseudo_train = torch.zeros(qtde_emb, dtype=bool)
    pseudo_train[pseudo_train_ix] = 1
    pyg_graph_dev.pseudo_train_mask = pseudo_train

    pseudo_val = torch.zeros(qtde_emb, dtype=bool)
    pseudo_val[pseudo_val_ix] = 1
    pyg_graph_dev.pseudo_val_mask = torch.zeros(qtde_emb, dtype=bool)

    aug_event_mask = torch.zeros(qtde_emb, dtype=bool)
    aug_event_mask[((pyg_graph_dev["joint"].y != 0) & (pyg_graph_dev["joint"].y != 1))] = 1
    pyg_graph_dev.aug_event_mask = aug_event_mask

    test_mask = torch.ones(qtde_emb, dtype=bool)
    test_mask[pyg_graph_dev.labeled_mask] = 0
    test_mask[pyg_graph_dev.unlbl_mask] = 0
    test_mask[pyg_graph_dev.aug_event_mask] = 0
    pyg_graph_dev.test_mask = test_mask

    return pyg_graph_train, pyg_graph_dev


def generate_graph(**kwargs):
    event = kwargs.get("event")
    reduction = kwargs.get("reduction")
    al_isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    cache = kwargs.get("use_cache")
    n_neigh_train = kwargs.get("n_neigh_train")
    n_neigh_full = kwargs.get("n_neigh_full")
    graph_mode = kwargs.get("graph_mode")
    imageft = kwargs.get("imageft")
    textft = kwargs.get("textft")
    aug_unlbl_set = kwargs.get("aug_unlbl_set")
    device = kwargs.get("device")

    if aug_unlbl_set == "one":
        aug_event = EVENT_AUG_PAIRS[event]
        filepath = AUG_GRAPH_CACHE.format(event, aug_event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    if aug_unlbl_set == "all":
        filepath = AUG_GRAPH_CACHE.format(event, "all", imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    if aug_unlbl_set == "all_differ":
        filepath = AUG_GRAPH_CACHE.format(event, "all_differ", imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    else:
        filepath = GRAPH_CACHE.format(event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)

    print("Feature Extraction - Text")
    if textft == "mpnet":
        [df_text_train, df_text_dev, _] = mpnet_features(**kwargs)
    elif textft == "clip":
        [df_text_train, df_text_dev, _] = clip_features(mode="text", **kwargs)

    print("Feature Extraction - Image")
    if imageft == "maxvit":
        [df_image_train, df_image_dev, _] = maxvit_features(**kwargs)
    elif textft == "clip":
        [df_image_train, df_image_dev, _] = clip_features(mode="image", **kwargs)

    data_train = pd.read_json(TRAIN_SET.format(event), lines=True)
    ft_train_images, ft_train_text, annot_train, _, _ = process_dataframe(data_train,
                                                                    df_image_train,
                                                                    df_text_train)

    # dev
    data_dev = pd.read_json(DEV_SET.format(event), lines=True)
    ft_dev_images, ft_dev_text, annot_dev, _, _ = process_dataframe(data_dev, df_image_dev, df_text_dev)

    if aug_unlbl_set is not None:
        annot_train = np.column_stack([annot_train, np.zeros(annot_train.shape[0])])

        if aug_unlbl_set == "one":
            aug_event = [EVENT_AUG_PAIRS[event]]
        elif aug_unlbl_set == "all_differ":
            aug_event = EVENT_AUG_DIFF[event]
        elif aug_unlbl_set == "all":
            aug_event = [e for e in EVENTS if e != event]

        for ix, ae in enumerate(aug_event):

            if textft == "mpnet":
                [df_text_aug, _, _] = mpnet_features(event_features=ae, **kwargs)
            elif textft == "clip":
                [df_text_aug, _, _] = clip_features(event_features=ae, mode="text", cache_features=True, **kwargs)
        
            if imageft == "maxvit":
                [df_image_aug, _, _] = maxvit_features(event_features=ae, **kwargs)
            elif textft == "clip":
                [df_image_aug, _, _] = clip_features(event_features=ae, mode="image", cache_features=True, **kwargs)
            
            df_image_aug["labels"] = 2
            df_text_aug["labels"] = 2

            data_aug = pd.read_json(TRAIN_SET.format(ae), lines=True)
            ft_aug_images_e, ft_aug_text_e, annot_aug_e, _, _ = process_dataframe(data_aug, df_image_aug, df_text_aug)

            if ix == 0:
                ft_aug_images = ft_aug_images_e
                ft_aug_text = ft_aug_text_e
                annot_aug = annot_aug_e
            else:
                ft_aug_images = torch.concat([ft_aug_images, ft_aug_images_e])
                ft_aug_text = torch.concat([ft_aug_text, ft_aug_text_e])
                annot_aug = np.concatenate([annot_aug, annot_aug_e])

            # sample
        if aug_unlbl_set in ("all_differ", "all"):
            if annot_train.shape[0] < annot_aug.shape[0]:
                sample_size = annot_train.shape[0]
                indices = torch.randperm(annot_aug.shape[0])[:sample_size]
                
                ft_aug_images = ft_aug_images[indices]
                ft_aug_text = ft_aug_text[indices]
                annot_aug = annot_aug[indices]

        annot_aug = np.zeros_like(annot_aug)
        annot_aug = np.column_stack([annot_aug, np.ones(annot_aug.shape[0])])

        ft_train_images = torch.concat([ft_train_images, ft_aug_images])
        ft_train_text = torch.concat([ft_train_text, ft_aug_text])
        annot_train = np.concatenate([annot_train, annot_aug])
    
    if reduction == "autoenc":
        autoenc = kwargs.get("autoenc")

        ft_train_images, ft_dev_images = autoencoder_reduction(autoenc,
                                                                ft_train_images,
                                                                ft_dev_images,
                                                                device,
                                                                "maxvit",
                                                                event)
        
        ft_train_text, ft_dev_text = autoencoder_reduction(autoenc,
                                                            ft_train_text,
                                                            ft_dev_text,
                                                            device,
                                                            "mpnet",
                                                            event)
    
    print(AL_SPLIT_SET.format(event, al_isel, "labeled", labeled_size, set_id))
    ft_mt_training_step = torch.concat([ft_train_images, ft_train_text], axis=1)
    
    ft_dev = torch.concat([ft_dev_images, ft_dev_text], axis=1)
    
    ft_mt_training_step = check_and_convert_to_tensor(ft_mt_training_step)
    annot_mt_training_step = check_and_convert_to_tensor(annot_train)
    annot_dev = check_and_convert_to_tensor(annot_dev)
    
    annot_mt_training_step = torch.argmax(annot_mt_training_step, dim=1)
    annot_dev = torch.argmax(annot_dev, dim=1)

    # train
    print("Generating Graph - Train")

    emb = ft_mt_training_step
    lbl = annot_mt_training_step

    edges = graph_edges(emb, n_neigh_train, graph_mode)

    pyg_graph_train = Data(x=emb, edge_index=edges, y=lbl)

    labeled_ix, unlabeled_ix, pseudo_train_ix, pseudo_val_ix = data_split(pyg_graph_train, **kwargs)
    qtde_emb = emb.shape[0]

    labeled_mask = torch.zeros(qtde_emb, dtype=bool)
    labeled_mask[labeled_ix] = 1
    pyg_graph_train.labeled_mask = labeled_mask

    unlabeled_mask = torch.zeros(qtde_emb, dtype=bool)
    unlabeled_mask[unlabeled_ix] = 1
    pyg_graph_train.unlbl_mask = unlabeled_mask

    pseudo_train = torch.zeros(qtde_emb, dtype=bool)
    pseudo_train[pseudo_train_ix] = 1
    pyg_graph_train.pseudo_train_mask = pseudo_train

    pseudo_val = torch.zeros(qtde_emb, dtype=bool)
    pseudo_val[pseudo_val_ix] = 1
    pyg_graph_train.pseudo_val_mask = pseudo_val

    aug_event_mask = torch.ones(qtde_emb, dtype=bool)
    aug_event_mask[((pyg_graph_train.y == 0) | (pyg_graph_train.y == 1))] = 0
    pyg_graph_train.aug_event_mask = aug_event_mask

    pyg_graph_train.to(device)

    # dev
    print("Generating Graph - Dev")

    ft_mt_dev_step = torch.concat([ft_mt_training_step, ft_dev])
    annot_mt_dev_step = torch.concat([annot_mt_training_step, annot_dev])

    emb = ft_mt_dev_step
    lbl = annot_mt_dev_step

    qtde_emb = emb.shape[0]

    edges = graph_edges(emb, n_neigh_full, graph_mode)

    pyg_graph_dev = Data(x=emb, edge_index=edges, y=lbl)

    labeled_mask = torch.zeros(qtde_emb, dtype=bool)
    labeled_mask[labeled_ix] = 1
    pyg_graph_dev.labeled_mask = labeled_mask

    unlabeled_mask = torch.zeros(qtde_emb, dtype=bool)
    unlabeled_mask[unlabeled_ix] = 1
    pyg_graph_dev.unlbl_mask = unlabeled_mask

    pseudo_train = torch.zeros(qtde_emb, dtype=bool)
    pseudo_train[pseudo_train_ix] = 1
    pyg_graph_dev.pseudo_train_mask = pseudo_train

    pseudo_val = torch.zeros(qtde_emb, dtype=bool)
    pseudo_val[pseudo_val_ix] = 1
    pyg_graph_dev.pseudo_val_mask = torch.zeros(qtde_emb, dtype=bool)

    aug_event_mask = torch.zeros(qtde_emb, dtype=bool)
    aug_event_mask[((pyg_graph_dev.y != 0) & (pyg_graph_dev.y != 1))] = 1
    pyg_graph_dev.aug_event_mask = aug_event_mask

    test_mask = torch.ones(qtde_emb, dtype=bool)
    test_mask[pyg_graph_dev.labeled_mask] = 0
    test_mask[pyg_graph_dev.unlbl_mask] = 0
    test_mask[pyg_graph_dev.aug_event_mask] = 0
    pyg_graph_dev.test_mask = test_mask

    G = nx.Graph()
    G.add_nodes_from(range(pyg_graph_train.num_nodes))
    edges = pyg_graph_train.edge_index.t().tolist()
    G.add_edges_from(edges)
    # betweenness = nx.betweenness_centrality(G)
    # pyg_graph_train.betweenness = torch.Tensor([betweenness[i] for i in range(pyg_graph_train.num_nodes)])

    pyg_graph_dev.to(device)

    return pyg_graph_train, pyg_graph_dev


def train_step(model, data, **kwargs):

    model.train()

    lr = float(kwargs.get('lr'))
    loss_func = kwargs.get('loss')
    wd = float(kwargs.get('weight_decay'))
    aug_unlbl_set = kwargs.get("aug_unlbl_set")
    dataset_loss = kwargs.get("dataset_loss")
    hetero = kwargs.get("hetero")
    sml = kwargs.get("single_modal_loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer.zero_grad()

    if hetero is True:
        out_full = model(data.x_dict, data.edge_index_dict)
        lbl = data["joint"].y
        lbl_image = data["image"].y
        lbl_text = data["text"].y
    else:
        out_full = model(data.x, data.edge_index)
        lbl = data.y

    if bool(sml) is True:
        [out_full, out_full_image, out_full_text, out_full_concat] = out_full
        
    if len(out_full.shape) == 3 and out_full.shape[1] == 1:
        out_full = out_full.squeeze()

    if aug_unlbl_set is not None and dataset_loss is True:
        out = out_full[data.pseudo_train_mask][:,:2]
        out_dataset = out_full[:,2:]

        res = lbl[(data.pseudo_train_mask)].long()
        res_dataset = data.aug_event_mask.long()
    else:
        out = out_full[(data.pseudo_train_mask) | (data.aug_event_mask)]
        res = lbl[(data.pseudo_train_mask) | (data.aug_event_mask)].long()

    if bool(sml) is True:
        # order image and text features to be like joint
        out_full_image = out_full_image[data.edge_index_dict[("image", "ref", "joint")][0]]
        out_full_text = out_full_text[data.edge_index_dict[("text", "ref", "joint")][0]]
        lbl_image = lbl_image[data.edge_index_dict[("image", "ref", "joint")][0]]
        lbl_text = lbl_text[data.edge_index_dict[("text", "ref", "joint")][0]]

        if len(out_full_image.shape) == 3 and out_full_image.shape[1] == 1:
            out_full_image = out_full_image.squeeze()
    
        if aug_unlbl_set is not None and dataset_loss is True:
            out_image = out_full_image[data.pseudo_train_mask][:,:2]
            out_image_dataset = out_full_image[:,2:]
    
            res_image = lbl_image[(data.pseudo_train_mask)].long()
            res_image_dataset = data.aug_event_mask.long()
        else:
            out_image = out_full_image[(data.pseudo_train_mask) | (data.aug_event_mask)]
            res_image = lbl_image[(data.pseudo_train_mask) | (data.aug_event_mask)].long()
        
        if len(out_full_text.shape) == 3 and out_full_text.shape[1] == 1:
            out_full_text = out_full_text.squeeze()
    
        if aug_unlbl_set is not None and dataset_loss is True:
            out_text = out_full_text[data.pseudo_train_mask][:,:2]
            out_text_dataset = out_full_text[:,2:]
    
            res_text = lbl_text[(data.pseudo_train_mask)].long()
            res_text_dataset = data.aug_event_mask.long()
        else:
            out_text = out_full_text[(data.pseudo_train_mask) | (data.aug_event_mask)]
            res_text = lbl_text[(data.pseudo_train_mask) | (data.aug_event_mask)].long()

        if aug_unlbl_set is not None and dataset_loss is True:
            out_concat = out_full_concat[data.pseudo_train_mask][:,:2]
            out_concat_dataset = out_full_concat[:,2:]
    
            res_concat = lbl[(data.pseudo_train_mask)].long()
            res_concat_dataset = data.aug_event_mask.long()
        else:
            out_concat = out_full_text[(data.pseudo_train_mask) | (data.aug_event_mask)]
            res_concat = lbl[(data.pseudo_train_mask) | (data.aug_event_mask)].long()

    if loss_func == "nll":
        loss = F.nll_loss(out, res)

        if bool(sml) is True:
            loss += F.nll_loss(out_image, res_image)
            loss += F.nll_loss(out_text, res_text)
            loss += F.nll_loss(out_concat, res_concat)
    
    elif loss_func == "nll_balanced":
        
        class_counts = torch.bincount(res)

        total_samples = len(res)
        class_weights = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])

        class_weights = class_weights.to(res.device)

        if out.shape[1] == class_weights.shape[0]:
            loss = F.nll_loss(out, res, weight=class_weights)
        else:
            loss = F.nll_loss(out, res)

        if bool(sml) is True:
            class_counts = torch.bincount(res_image)
    
            total_samples = len(res_image)
            class_weights_image = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])
    
            class_weights_image = class_weights_image.to(res.device)

            if out_image.shape[1] == class_weights_image.shape[0]:
                loss += F.nll_loss(out_image, res_image, weight=class_weights_image)
            else:
                loss += F.nll_loss(out_image, res_image)

            class_counts = torch.bincount(res_text)
    
            total_samples = len(res_text)
            class_weights_text = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])
    
            class_weights_text = class_weights_text.to(res.device)

            if out_text.shape[1] == class_weights_text.shape[0]:
                loss += F.nll_loss(out_text, res_text, weight=class_weights_text)
            else:
                loss += F.nll_loss(out_text, res_text)

            if out.shape[1] == class_weights.shape[0]:
                loss += F.nll_loss(out_concat, res_concat, weight=class_weights)
            else:
                loss += F.nll_loss(out_concat, res_concat)

    if aug_unlbl_set is not None and dataset_loss is True:
        class_counts = torch.bincount(res_dataset)

        total_samples = len(res_dataset)
        class_weights_dataset = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])

        class_weights_dataset = class_weights.to(res_dataset.device)

        loss += F.nll_loss(out_dataset, res_dataset, weight=class_weights_dataset)

        if sml is True:
            class_counts = torch.bincount(res_image_dataset)
    
            total_samples = len(res_image_dataset)
            class_weights_image_dataset = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])
    
            class_weights_image_dataset = class_weights_image.to(res.device)
            loss += F.nll_loss(out_image, res_image, weight=class_weights_image_dataset)

            class_counts = torch.bincount(res_text_dataset)
    
            total_samples = len(res_text_dataset)
            class_weights_text_dataset = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])
    
            class_weights_text_dataset = class_weights_text.to(res.device)
            loss += F.nll_loss(out_text, res_text, weight=class_weights_text_dataset)

            loss += F.nll_loss(out_concat_dataset, res_concat_dataset, weight=class_weights_dataset)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def eval_data(model, data, test=False, train_val=False, result_file=None, **kwargs):
    hetero = kwargs.get("hetero")
    num_test_inference_run = kwargs.get("num_test_inference_run")
    sml = kwargs.get("single_modal_loss")

    model.eval()

    if hetero is True:
        preds = model(data.x_dict, data.edge_index_dict, num_test_inference_run)

        if bool(sml) is True:
            preds = preds[-1]
    else:
        preds = model(data.x, data.edge_index, num_test_inference_run)

    if len(preds.shape) == 3:
        preds = torch.logsumexp(preds, dim=1) - math.log(preds.shape[1])

    if hetero is True:
        lbl = data["joint"].y
    else:
        lbl = data.y

    if train_val:
        mask_train = data.pseudo_train_mask
        pred_train = preds[mask_train].max(1)[1]
        f1_train = get_f1(lbl[mask_train], pred_train)
    
        mask_val = data.pseudo_val_mask
        pred_val = preds[mask_val].max(1)[1]
        f1_val = get_f1(lbl[mask_val], pred_val)
    else:
        mask_labeled = data.labeled_mask
        pred_labeled = preds[mask_labeled].max(1)[1]
        f1_labeled = get_f1(lbl[mask_labeled], pred_labeled)
        bacc_labeled = get_normalized_acc(lbl[mask_labeled], pred_labeled)

    if test is True:
        mask_unlbl = data.unlbl_mask
        pred_unlbl = preds[mask_unlbl].max(1)[1]
        f1_unlbl = get_f1(lbl[mask_unlbl], pred_unlbl)
        bacc_unlbl = get_normalized_acc(lbl[mask_unlbl], pred_unlbl)

        mask_test = data.test_mask
        pred_test = preds[mask_test].max(1)[1]
        f1_test = get_f1(lbl[mask_test], pred_test)
        bacc_test = get_normalized_acc(lbl[mask_test], pred_test)
        confm = confusion_matrix(lbl[mask_test].cpu(), pred_test.cpu())

        if result_file is not None:
            results = dict()
            results["annot"] = lbl[mask_test].cpu().numpy().tolist()
            results["pred"] = pred_test.cpu().numpy().tolist()
            results["f1_labeled"] = float(f1_labeled)
            results["bacc_labeled"] = float(bacc_labeled)
            results["f1_unlbl"] = float(f1_unlbl)
            results["bacc_unlbl"] = float(bacc_unlbl)
            results["f1_test"] = float(f1_test)
            results["bacc_test"] = float(bacc_test)

            with open(result_file, "w") as fp:
                json.dump(results, fp, indent=4, default=custom_serializer)

            # Plot the confusion matrix using seaborn
            plt.figure(figsize=(5, 5))
            sns.heatmap(confm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(result_file.replace(".json", ".png"))
        
        # log_post_table(torch.where(mask_test == True)[0], pred_test, 
        #                data.y[mask_test], preds[mask_test], **kwargs)

        return f1_labeled, f1_unlbl, f1_test
    elif train_val == True :
        bacc_test = None
        return f1_train, f1_val, preds


def run_base(model, pyg_graph, **kwargs):
    hetero = kwargs.get("hetero")

    model.reset_parameters()

    if hetero is True:
        y = pyg_graph["joint"].y
    else:
        y = pyg_graph.y.clone()

    y = y.to("cpu")
    
    best_model = None
    best_score = -1 * torch.inf

    epochs = kwargs.get('epochs')

    model.train()

    with tqdm.trange(epochs, unit="epoch", mininterval=0, position=0, leave=True ) as bar:
        epoch = 0
        best_epoch = -1
        early_stopping_counter = 0
        while True:
            bar.set_description(f"Epoch {epoch+1}")

            loss = train_step(model, pyg_graph, **kwargs)

            train_f1, val_f1, preds = eval_data(model, pyg_graph, train_val=True, **kwargs)

            if kwargs.get("best_model_metric") == "best_val":
                epoch_score = val_f1
            elif kwargs.get("best_model_metric") == "best_hm":
                if train_f1 > 0 and val_f1 > 0:
                    epoch_score = statistics.harmonic_mean([train_f1, val_f1])
                else:
                    epoch_score = 0

            if epoch_score > best_score:
                best_model = copy.deepcopy(model)
                best_score = epoch_score
                best_epoch = epoch
                
                early_stopping_counter = 0 
            else:
                early_stopping_counter += 1
            
            
            bar.update(1)
            bar.set_postfix(
                loss=float(loss),
                f1_train=train_f1,
                f1_val=val_f1
            )

            epoch += 1

            if epoch == epochs:
                break
            elif early_stopping_counter == kwargs.get("early_stopping") and best_epoch > 0:
                break
            elif epoch_score == 1:
                break

    if epoch == epochs:
        print(f"End of training")
    if early_stopping_counter == kwargs.get("early_stopping") and best_epoch > 0:
        print(f"Early stopping at epoch {epoch}. Validation loss did not improve.")
    elif epoch_score == 1:
        print(f"Early stopping at epoch {epoch}. Metric for best model equals to 1")
    
    return best_model


def validate_best_model(best_model, pyg_graph_test, result_file=None, **kwargs):
    display = True
    print(result_file)

    labeled_f1, unlabeled_f1, test_f1 = eval_data(best_model, pyg_graph_test, test=True, result_file=result_file, **kwargs)

    if display is True:
        print("---------------------")
        print("Best Model (FULL GRAPH):")
        print(f'Labeled F1: {100 * labeled_f1:.2f}%, '
            f'Unlabeled F1: {100 * unlabeled_f1:.2f}% '
            f'Test F1: {100 * test_f1:.2f}%')
        print("---------------------")

    return test_f1


def get_arch(**kwargs):
    group = kwargs.get("group")
    hetero = kwargs.get("hetero")