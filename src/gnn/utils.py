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
import igraph as ig

from src.al.isel import data_split
from src.feature_extraction.feature_extraction import clip_features, mpnet_features, maxvit_features
from src.utils.utils import check_and_convert_to_tensor, custom_serializer, get_f1, get_normalized_acc, process_dataframe
from src.utils.constants import AL_SPLIT_SET, AUG_GRAPH_CACHE, DEV_SET, GRAPH_CACHE, TRAIN_SET, EVENT_AUG_PAIRS
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

    if aug_unlbl_set is True:
        aug_event = EVENT_AUG_PAIRS[event]
        filepath = AUG_GRAPH_CACHE.format(event, aug_event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)
    else:
        filepath = GRAPH_CACHE.format(event, imageft, textft, reduction, graph_mode, al_isel, n_neigh_train, n_neigh_full, labeled_size, set_id)

    if os.path.exists(filepath) and cache:
        with open(filepath, 'rb') as fp:
            pyg_graph_train, pyg_graph_dev = pickle.load(fp)

            return pyg_graph_train, pyg_graph_dev

    if textft == "mpnet":
        [df_text_train, df_text_dev, _] = mpnet_features(**kwargs)
    elif textft == "clip":
        [df_text_train, df_text_dev, _] = clip_features(mode="text", **kwargs)

    if imageft == "maxvit":
        [df_image_train, df_image_dev, _] = maxvit_features(**kwargs)
    elif textft == "clip":
        [df_image_train, df_image_dev, _] = clip_features(mode="image", **kwargs)

    data_train = pd.read_json(TRAIN_SET.format(event), lines=True)
    ft_train_images, ft_train_text, annot_train = process_dataframe(data_train,
                                                                    df_image_train,
                                                                    df_text_train)

    # dev
    data_dev = pd.read_json(DEV_SET.format(event), lines=True)
    ft_dev_images, ft_dev_text, annot_dev = process_dataframe(data_dev, df_image_dev, df_text_dev)

    if aug_unlbl_set is True:
        aug_event = EVENT_AUG_PAIRS[event]
        if textft == "mpnet":
            [df_text_aug, _, _] = mpnet_features(event_features=aug_event, **kwargs)
        elif textft == "clip":
            [df_text_aug, _, _] = clip_features(event_features=aug_event, mode="text", **kwargs)
    
        if imageft == "maxvit":
            [df_image_aug, _, _] = maxvit_features(event_features=aug_event, **kwargs)
        elif textft == "clip":
            [df_image_aug, _, _] = clip_features(event_features=aug_event, mode="image", **kwargs)

        df_image_aug["labels"] = 2
        df_text_aug["labels"] = 2

        data_aug = pd.read_json(TRAIN_SET.format(aug_event), lines=True)
        ft_aug_images, ft_aug_text, annot_aug = process_dataframe(data_aug, df_image_aug, df_text_aug)

        annot_train = np.column_stack([annot_train, np.zeros(annot_train.shape[0])])
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

    # dev
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

    with open(filepath, "wb") as fp:
        pyg_graph_dev.to("cpu")
        pyg_graph_train.to("cpu")

        pickle.dump([pyg_graph_train, pyg_graph_dev], fp)     

        pyg_graph_dev.to(device)
        pyg_graph_train.to(device)
    
    return pyg_graph_train, pyg_graph_dev


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
    out = out_full[(data.pseudo_train_mask) | (data.aug_event_mask)]
        
    if len(out.shape) == 3 and out.shape[1] == 1:
        out = out.squeeze()
    
    if loss_func == 'nll':
        res = data.y[(data.pseudo_train_mask) | (data.aug_event_mask)].long()

        loss = F.nll_loss(out, res)
    
    elif loss_func == "nll_balanced":
        res = data.y[(data.pseudo_train_mask) | (data.aug_event_mask)].long()
        
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
    aug_unlbl_set = kwargs.get("aug_unlbl_set")

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
        
        # log_post_table(torch.where(mask_test == True)[0], pred_test, 
        #                data.y[mask_test], preds[mask_test], **kwargs)

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

    wandb = kwargs.get("wandb")

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

                  # 🐝 2️⃣ Log metrics from your script to W&B
            # wandb.log({"f1_train": train_f1,
            #                          "f1_val": val_f1, 
            #                          "loss": loss})

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

    wandb = kwargs.get('wandb')

    labeled_f1, unlabeled_f1, test_f1 = eval_data(best_model, pyg_graph_test, test=True, result_file=result_file, **kwargs)

    wandb.summary["labeled_f1"] = labeled_f1
    wandb.summary["unlabeled_f1"] = unlabeled_f1
    wandb.summary["test_f1"] = test_f1

    if display is True:
        print("---------------------")
        print("Best Model (FULL GRAPH):")
        print(f'Labeled F1: {100 * labeled_f1:.2f}%, '
            f'Unlabeled F1: {100 * unlabeled_f1:.2f}% '
            f'Test F1: {100 * test_f1:.2f}%')
        print("---------------------")

    return test_f1

def log_post_table(id_post, predicted, labels, probs, **kwargs):
    wandb = kwargs.get("wandb")

    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for ix, pred, targ, prob in zip(id_post.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(ix, pred, targ, *prob.numpy()))
    wandb.log({"predictions_table":table}, commit=False)