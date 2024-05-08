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
from torch_geometric.data import Data
import tqdm
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
from torch_geometric.utils import degree

from src.feature_extraction.feature_extraction import mpnet_features, maxvit_features
from src.utils.utils import check_and_convert_to_tensor, custom_serializer, data_split, get_f1, get_normalized_acc, process_dataframe
from src.utils.constants import AL_SPLIT_SET, DEV_SET, GRAPH_CACHE, TRAIN_SET
from src.utils.reduction import autoencoder_reduction


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

    


def graph_edges(emb, n_neighbors):

    if torch.is_tensor(emb) is True:
        device = emb.device
        emb = emb.detach().cpu().numpy()
    
    simm = cosine_similarity(emb)
    emb = torch.Tensor(emb).to(device)

    simm[np.arange(simm.shape[0]),np.arange(simm.shape[0])] = 0
    
    edges = set()        
    for i, vec in enumerate(simm):
        partit = np.argpartition(vec, -1*n_neighbors)
        for j in range(n_neighbors):
            edges.add((i, partit[-1 * j]))
            edges.add((partit[-1 * j], i))
        
    edges = torch.Tensor(list(edges)).t().type(torch.int64)
    
    return edges
    

def generate_graph(**kwargs):
    event = kwargs.get("event")
    dev_id = kwargs.get("device")
    reduction = kwargs.get("reduction")
    al_isel = kwargs.get("al_isel")
    labeled_size = kwargs.get("labeled_size")
    set_id = kwargs.get("set_id")
    cache = kwargs.get("use_cache")

    filepath = GRAPH_CACHE.format(event, reduction, al_isel, labeled_size, set_id)

    if os.path.exists(filepath) and cache:
        with open(filepath, 'rb') as fp:
            pyg_graph_train, pyg_graph_dev = pickle.load(fp)

            return pyg_graph_train, pyg_graph_dev

    n_neigh_train = kwargs.get("n_neigh_train")
    n_neigh_full = kwargs.get("n_neigh_full")

    if dev_id is not None:
        device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    [df_text_train, df_text_dev, _] = mpnet_features(dev_id, event)
    [df_image_train, df_image_dev, _] = maxvit_features(dev_id, event)
    
    
    data_train = pd.read_json(TRAIN_SET.format(event), lines=True)
    ft_images_train, ft_text_train, annot_train = process_dataframe(data_train,
                                                    df_image_train,
                                                    df_text_train)
    
    # dev
    data_dev = pd.read_json(DEV_SET.format(event), lines=True)
    ft_dev_images, ft_dev_text, annot_dev = process_dataframe(data_dev, df_image_dev, df_text_dev)
    
    if reduction == "autoenc":
        autoenc = kwargs.get("autoenc")

        ft_train_images, ft_dev_images = autoencoder_reduction(autoenc,
                                                                ft_images_train,
                                                                ft_dev_images,
                                                                device,
                                                                "maxvit",
                                                                event)
        
        ft_train_text, ft_dev_text = autoencoder_reduction(autoenc,
                                                            ft_text_train,
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
    emb = ft_mt_training_step
    lbl = annot_mt_training_step

    edges = graph_edges(emb, n_neigh_train)

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
    print(pseudo_val_ix)
    pseudo_val[pseudo_val_ix] = 1
    pyg_graph_train.pseudo_val_mask = pseudo_val

    # dev
    ft_mt_dev_step = torch.concat([ft_mt_training_step, ft_dev])
    annot_mt_dev_step = torch.concat([annot_mt_training_step, annot_dev])

    emb = ft_mt_dev_step
    lbl = annot_mt_dev_step

    qtde_emb = emb.shape[0]

    edges = graph_edges(emb, n_neigh_full)

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

    test_mask = torch.ones(qtde_emb, dtype=bool)
    test_mask[pyg_graph_dev.labeled_mask] = 0
    test_mask[pyg_graph_dev.unlbl_mask] = 0
    pyg_graph_dev.test_mask = test_mask

    with open(filepath, "wb") as fp:
        pickle.dump([pyg_graph_train, pyg_graph_dev], fp)     
    
    return pyg_graph_train, pyg_graph_dev


def generate_graph_old(emb, lbl, n_neighbors, **kwargs):

    qtde_emb = emb.shape[0]

    if torch.is_tensor(emb) is True:
        device = emb.device
        emb = emb.detach().cpu().numpy()
    
    simm = cosine_similarity(emb)
    emb = torch.Tensor(emb).to(device)

    simm[np.arange(simm.shape[0]),np.arange(simm.shape[0])] = 0
    
    edges = set()        
    for i, vec in enumerate(simm):
        partit = np.argpartition(vec, -1*n_neighbors)
        for j in range(n_neighbors):
            edges.add((i, partit[-1 * j]))
            edges.add((partit[-1 * j], i))
        
    edges = torch.Tensor(list(edges)).t().type(torch.int64)

    pyg_graph = Data(x=emb, edge_index=edges, y=lbl)

    if kwargs.get("labeled_ix") is not None:
        labeled_mask = torch.zeros(qtde_emb, dtype=bool)
        labeled_mask[kwargs.get("labeled_ix")] = 1
        pyg_graph.labeled_mask = labeled_mask

    if kwargs.get("unlabeled_ix") is not None:
        unlabeled_mask = torch.zeros(qtde_emb, dtype=bool)
        unlabeled_mask[kwargs.get("unlabeled_ix")] = 1
        pyg_graph.unlbl_mask = unlabeled_mask

    if kwargs.get("test_ix") is not None:
        test_mask = torch.zeros(qtde_emb, dtype=bool)
        test_mask[kwargs.get("test_ix")] = 1
        pyg_graph.test_mask = test_mask
    
    if kwargs.get("event_unlbl_ix") is not None:
        event_unlbl_mask = torch.zeros(qtde_emb, dtype=bool)
        event_unlbl_mask[kwargs.get("event_unlbl_ix")] = 1
        pyg_graph.event_unlbl_mask = event_unlbl_mask
    
    return pyg_graph

    
def train_step(model, data, **kwargs):

    model.train()

    lr = float(kwargs.get('lr'))
    loss_func = kwargs.get('loss')
    wd = float(kwargs.get('weight_decay'))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer.zero_grad()

    if kwargs.get("edge_attr"):
        out_full = model(data.x, data.edge_index, data.edge_attr)
    else:
        out_full = model(data.x, data.edge_index)
    out = out_full[data.pseudo_train_mask]
        
    if len(out.shape) == 3 and out.shape[1] == 1:
        out = out.squeeze()
    
    if loss_func == 'nll':
        res = data.y[data.pseudo_train_mask].long()

        loss = F.nll_loss(out, res)
    
    elif loss_func == "nll_balanced":
        res = data.y[data.pseudo_train_mask].long()
        
        class_counts = torch.bincount(res)

        total_samples = len(res)
        class_weights = torch.Tensor([total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))])

        class_weights = class_weights.to(res.device)

        loss = F.nll_loss(out, res, weight=class_weights)
        
    elif loss_func == 'bce':
        res = torch.zeros_like(out)
        res[range(out.shape[0]), data.y[data.pseudo_train_mask].long()] = 1

        loss = torch.nn.BCEWithpredsLoss()(out, res)
    
    if kwargs.get('reg') == 'l2':
        l2_lambda = kwargs.get('l2_lambda')

        l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())
 
        loss = loss + float(l2_lambda) * l2_norm

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def eval_data(model, data, test=False, train_val=False, result_file=None, **kwargs):
    model.eval()

    preds = model(data.x, data.edge_index, kwargs.get("num_test_inference_run"))

    if len(preds.shape) == 3:
        preds = torch.logsumexp(preds, dim=1) - math.log(preds.shape[1])

    if train_val:
        mask_train = data.pseudo_train_mask
        pred_train = preds[mask_train].max(1)[1]
        f1_train = get_f1(data.y[mask_train], pred_train)
    
        mask_val = data.pseudo_val_mask
        pred_val = preds[mask_val].max(1)[1]
        f1_val = get_f1(data.y[mask_val], pred_val)
    else:
        mask_labeled = data.labeled_mask
        pred_labeled = preds[mask_labeled].max(1)[1]
        f1_labeled = get_f1(data.y[mask_labeled], pred_labeled)
        bacc_labeled = get_normalized_acc(data.y[mask_labeled], pred_labeled)

    if test is True:
        mask_unlbl = data.unlbl_mask
        pred_unlbl = preds[mask_unlbl].max(1)[1]
        f1_unlbl = get_f1(data.y[mask_unlbl], pred_unlbl)
        bacc_unlbl = get_normalized_acc(data.y[mask_unlbl], pred_unlbl)

        mask_test = data.test_mask
        pred_test = preds[mask_test].max(1)[1]
        f1_test = get_f1(data.y[mask_test], pred_test)
        bacc_test = get_normalized_acc(data.y[mask_test], pred_test)
        confm = confusion_matrix(data.y[mask_test].cpu(), pred_test.cpu())

        if result_file is not None:
            results = dict()
            results["annot"] = data.y[mask_test].cpu().numpy().tolist()
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

        return f1_labeled, f1_unlbl, f1_test
    elif train_val == True :
        bacc_test = None
        return f1_train, f1_val, preds


def run_base(model, pyg_graph, **kwargs):
    model.reset_parameters()

    n = pyg_graph.num_nodes

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
                best_preds = preds
                
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