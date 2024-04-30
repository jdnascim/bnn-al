import copy
import json
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
import yaml
import math

from src.utils.utils import custom_serializer, get_f1, get_normalized_acc

def generate_graph(emb, lbl, n_neighbors, **kwargs):

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
    out = out_full[data.train_mask]
        
    if len(out.shape) == 3 and out.shape[1] == 1:
        out = out.squeeze()
    
    if loss_func == 'nll':
        res = data.y[data.train_mask].long()
            
        loss = F.nll_loss(out, res)
        
    elif loss_func == 'bce':
        res = torch.zeros_like(out)
        res[range(out.shape[0]), data.y[data.train_mask].long()] = 1

        loss = torch.nn.BCEWithpredsLoss()(out, res)
    
    if kwargs.get('reg') == 'l2':
        l2_lambda = kwargs.get('l2_lambda')

        l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())
 
        loss = loss + l2_lambda * l2_norm

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
        mask_train = data.train_mask
        pred_train = preds[mask_train].max(1)[1]
        f1_train = get_f1(data.y[mask_train], pred_train)
    
        mask_val = data.val_mask
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
    best_score = 0

    epochs = kwargs.get('epochs')
    lbl_train_frac = kwargs.get('lbl_train_frac')

    labeled_ix = torch.argwhere(pyg_graph.labeled_mask).squeeze()
    if lbl_train_frac < 1:
    # split annotated data
        labeled_ix = labeled_ix.clone().to('cpu')
        lbl_train_ix, lbl_val_ix = train_test_split(labeled_ix, 
                                                train_size=lbl_train_frac,
                                                stratify=y[labeled_ix],
                                                random_state=0)
        
        pyg_graph.train_mask = torch.zeros(n, dtype=bool)
        pyg_graph.train_mask[lbl_train_ix] = True
        pyg_graph.val_mask = torch.zeros(n, dtype=bool)
        pyg_graph.val_mask[lbl_val_ix] = True
    elif lbl_train_frac == 1:
        pyg_graph.train_mask = torch.zeros(n, dtype=bool)
        pyg_graph.train_mask[labeled_ix] = True
        pyg_graph.val_mask = torch.zeros(n, dtype=bool)
        pyg_graph.val_mask[labeled_ix] = True


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
                epoch_score = statistics.harmonic_mean([train_f1, val_f1])

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

    return test_f1