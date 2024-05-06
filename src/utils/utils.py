import os
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, balanced_accuracy_score
from os.path import join
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree
import torch
import tqdm
import math

from src.feature_extraction.feature_extraction import IMAGEPATH
from src.utils.constants import TRAIN_SET

def get_normalized_acc(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()

    return balanced_accuracy_score(y_true, y_pred)


def get_f1(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu()

    if len(y_pred.shape) == 3:
        num_test_inference = y_pred.shape[1]
        y_pred = torch.logsumexp(y_pred, dim=1) - math.log(num_test_inference)

    return f1_score(y_true, y_pred, average='weighted')


def seed_everything(seed=42):
    """
    seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def check_and_convert_to_tensor(var):
    if not torch.is_tensor(var):
        return torch.Tensor(var)
    return var


def process_dataframe(data, df_image, df_text):
    df_image = df_image.set_index(['image_files','text'])
    df_text = df_text.set_index(['image_files','text'])
    
    df_image_dict = df_image.to_dict("index")
    df_text_dict = df_text.to_dict("index")

    image_files = []
    texts = []
    annotations = []
    for _, row in tqdm.tqdm(data.iterrows()):
        image_files.append(join(IMAGEPATH, row['image']))
        # Apply Preprocess stage here

        #texts.append(text_preprocess.pre_process(row['text'], keep_hashtag=True, keep_special_symbols=False))
        texts.append(row['text'])

        annotations.append(row['label'])

    annotations = [(1, 0) if l == 'not_informative' else (0, 1) for l in annotations]
    annotations = np.array(annotations)

    ft_images = torch.zeros([len(data), len(df_image["embeddings"].iloc[0])])
    for ix, (img, txt) in enumerate(zip(image_files, texts)):
        ft_images[ix] = torch.Tensor(df_image_dict[(img, txt)]["embeddings"])
    ft_text = torch.zeros([len(data), len(df_text["embeddings"].iloc[0])])
    for ix, (img, txt) in enumerate(zip(image_files, texts)):
        ft_text[ix] = torch.Tensor(df_text_dict[(img, txt)]["embeddings"])
    
    return ft_images, ft_text, annotations

    
def data_split_old(ft, labeled_size, set_id, isel="random"):

    qtde_items = ft.shape[0]

    if isel == "random":
        labeled_ix, unlbl_ix = train_test_split(np.arange(qtde_items), train_size=labeled_size, random_state=set_id)

    return labeled_ix, unlbl_ix


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


def custom_serializer(obj):
    if isinstance(obj, (list, tuple)):
        return ' '.join(str(x) for x in obj)
    else:
        return obj