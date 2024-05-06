import argparse
from datetime import datetime

from sklearn.model_selection import train_test_split
from src.al.al import bald_base, bald_deg_sel, bald_kmeans, bald_sel, batchbald_deg_sel, batchbald_sel, rdn_sel, unc_kmeans_sel, unc_sel
from src.utils.gnn import run_base, generate_graph, validate_best_model
import torch
import os
import torch.nn.functional as F
from src.arch.bnn import BayesianGNN
from src.arch.base import BaseMLP
from tqdm.auto import trange
from src.utils.constants import DEV_SET, RESULT_FILE, SETUP_FILE, AL_SPLIT_SET, TRAIN_SET
from src.utils.utils import seed_everything
import yaml
import numpy as np


start_time = datetime.now()


parser = argparse.ArgumentParser(description="Process command-line arguments")
parser.add_argument("--exp_id", type=int, required=True, help="Id of the experiment")
parser.add_argument("--exp_group", type=str, required=True, help="Group of the experiment")
parser.add_argument("--event", type=str, required=True, help="Name of the event")
parser.add_argument("--labeled_size", type=int, required=True, help="Size of labeled data")
parser.add_argument("--set_id", type=int, required=True, help="ID of the data set")
parser.add_argument("--arch", type=str, default="base_gnn", help="Architecture type")
parser.add_argument("--arch_setup", type=str, help="Architecture setup")
parser.add_argument("--device", type=str, default="cpu", help="Device for the experiment")
parser.add_argument("--run_id", default=0, type=int, required=False)
parser.add_argument("--reduction", default=None, type=str, required=False, choices=['pca', "autoenc"])
parser.add_argument("--autoenc", default="ae_base", type=str, required=False)
parser.add_argument("--pca_red", default=256, type=int, required=False)
parser.add_argument("--al", default=None, type=str, required=False)
parser.add_argument("--al_iter", default=3, type=int)
parser.add_argument("--al_batch", default=10, type=int)
parser.add_argument("--al_isel", default="random", type=str, required=False, choices=["random", "kmeans", "degree"])
parser.add_argument("--aug_unlbl_set", action="store_true", default=False)
parser.add_argument("--al_random_pseudo_val", action="store_true", default=False)
parser.add_argument("--retrain", action="store_true", default=False)
parser.add_argument("--use-cache", action="store_true", default=False)


args = parser.parse_args()

kwargs = vars(args)

exp_id = args.exp_id
exp_group = args.exp_group
event = args.event
labeled_size = args.labeled_size
set_id = args.set_id
arch = args.arch
dev_id = args.device
run_id = args.run_id
reduction = args.reduction
al = args.al
al_iter = args.al_iter
al_batch = args.al_batch
al_isel = args.al_isel

if args.aug_unlbl_set:
    aug_unlbl_set = True
else:
    aug_unlbl_set = False

if args.retrain:
    retrain = True
else:
    retrain = False

if args.al_random_pseudo_val:
    random_pseudo_val = True
else:
    random_pseudo_val = False

if args.arch_setup is not None:
    arch_setup = args.arch_setup
else:
    arch_setup = arch

with open(SETUP_FILE, "r") as yaml_file:
    arch_setup_data = yaml.safe_load(yaml_file)[arch_setup]

kwargs.update(arch_setup_data)

input_dim = arch_setup_data["input_dim"]
hidden_dim = arch_setup_data["hidden_dim"]
n_hidden= arch_setup_data["n_hidden"]
output_dim = arch_setup_data["output_dim"]

lbl_train_frac = arch_setup_data["lbl_train_frac"]
print(lbl_train_frac)

if dev_id is not None:
    device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

use_gpu = device != 'cpu'

seed_everything(set_id)

if arch_setup_data["group"] == "mlp":
    pyg_graph_dev = generate_graph(mode="dev", **kwargs)
    pyg_graph_dev = pyg_graph_dev.to(device)

    model = BaseMLP(input_dim, output_dim, hidden_dim, n_hidden)
    model = model.to(device)

    ft_labeled = pyg_graph_dev.x[pyg_graph_dev.pseudo_train]
    annot_labeled = pyg_graph_dev.x[pyg_graph_dev.pseudo_train]

    model.train()

    num_iterations = 20000
    progress_bar = trange(num_iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    optimizer.zero_grad()
    
    for j in progress_bar:
        out = model(ft_labeled)
        loss = F.nll_loss(out, annot_labeled)
        
        loss.backward()
        optimizer.step()
        progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(ft_labeled)))

    model.eval()
    ft_dev = pyg_graph_dev.x[pyg_graph_dev.test_mask]
    y_pred = model(ft_dev)
    bin_pred = np.argmax(y_pred.detach().cpu().numpy(),axis=1)
    
elif arch_setup_data["group"] == "gnn":

    model = BayesianGNN()
    model = model.to(device)

    pyg_graph_train, pyg_graph_dev = generate_graph(**kwargs)
    pyg_graph_train = pyg_graph_train.to(device)
    pyg_graph_dev = pyg_graph_dev.to(device)

    print("instance selection - train")
    print("ix: ", torch.arange(pyg_graph_train.x.shape[0], device=device)[pyg_graph_train.pseudo_train_mask])
    print("y: ", pyg_graph_train.y[pyg_graph_train.pseudo_train_mask])
    print("instance selection - val")
    print("ix: ", torch.arange(pyg_graph_train.x.shape[0], device=device)[pyg_graph_train.pseudo_val_mask])
    print("y: ", pyg_graph_train.y[pyg_graph_train.pseudo_val_mask])
    print(torch.unique(pyg_graph_train.y, return_counts=True))

    for i in range(args.al_iter + 1):
        
        best_model = run_base(model, pyg_graph_train, **kwargs)

        resf = RESULT_FILE.format(event, exp_group, exp_id, labeled_size, set_id, run_id)
        os.makedirs(resf[:resf.rfind("/")], exist_ok=True)

        validate_best_model(best_model, pyg_graph_dev, resf, **kwargs)

        # Active Learning
        if args.al is not None and i < al_iter:
            print("AL Iteration: {} of {}".format(i+1, al_iter))
            model = best_model

            if random_pseudo_val:
                al_batch_pseudo_train = int(al_batch * lbl_train_frac)
                al_batch_pseudo_val = al_batch - al_batch_pseudo_train
            else:
                al_batch_pseudo_train = al_batch
                
            if al == "random":
                selected_indices = rdn_sel(pyg_graph_train, al_batch_pseudo_train)
                
            elif al == "unc":
                selected_indices = unc_sel(model, pyg_graph_train, al_batch_pseudo_train)
              
            elif al == "unc-kmeans":
                selected_indices = unc_kmeans_sel(model, pyg_graph_train, al_batch_pseudo_train)

            elif al == "bald-base":
                selected_indices = bald_base(model, pyg_graph_train, al_batch_pseudo_train, bald_iter=al_batch_pseudo_train)
                
            elif al == "bald-kmeans":
                selected_indices = bald_kmeans(model, pyg_graph_train, al_batch_pseudo_train, bald_iter=al_batch_pseudo_train)
            
            elif al == "batchbald":
                selected_indices = batchbald_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
                selected_indices = selected_indices.squeeze()

            elif al == "bald":
                selected_indices = bald_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
                selected_indices = selected_indices.squeeze()

            elif al == "batchbald-degree":
                selected_indices = batchbald_deg_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
                selected_indices = selected_indices.squeeze()

            elif al == "bald-degree":
                selected_indices = bald_deg_sel(model, pyg_graph_train, al_batch_pseudo_train, device, bald_iter=al_batch_pseudo_train)
                selected_indices = selected_indices.squeeze()

            pyg_graph_train.labeled_mask[selected_indices] = True
            pyg_graph_train.unlbl_mask[selected_indices] = False
            pyg_graph_dev.labeled_mask[selected_indices] = True
            pyg_graph_dev.unlbl_mask[selected_indices] = False

            if random_pseudo_val:
                selected_indices_pseudo_val = rdn_sel(pyg_graph_train, al_batch_pseudo_val)
                pyg_graph_train.pseudo_train_mask[selected_indices] = True
                pyg_graph_train.pseudo_val_mask[selected_indices_pseudo_val] = True
                pyg_graph_train.labeled_mask[selected_indices_pseudo_val] = True
                pyg_graph_train.unlbl_mask[selected_indices_pseudo_val] = False
                pyg_graph_dev.pseudo_train_mask[selected_indices] = True
                pyg_graph_dev.pseudo_val_mask[selected_indices_pseudo_val] = True
                pyg_graph_dev.labeled_mask[selected_indices_pseudo_val] = True
                pyg_graph_dev.unlbl_mask[selected_indices_pseudo_val] = False

                labeled_size += len(selected_indices) + len(selected_indices_pseudo_val)
            else:
                pseudo_train, pseudo_val = train_test_split(selected_indices, train_size=lbl_train_frac, random_state=set_id)
                pyg_graph_train.pseudo_train_mask[pseudo_train] = True
                pyg_graph_train.pseudo_val_mask[pseudo_val] = True
                pyg_graph_dev.pseudo_train_mask[pseudo_train] = True
                pyg_graph_dev.pseudo_val_mask[pseudo_val] = True

                labeled_size += len(selected_indices)
                
            print("new train set")
            print("ix: ", torch.arange(pyg_graph_train.x.shape[0], device=device)[pyg_graph_train.pseudo_train_mask])
            print("y: ", pyg_graph_train.y[pyg_graph_train.pseudo_train_mask])
            print("new val set")
            print("ix: ", torch.arange(pyg_graph_train.x.shape[0], device=device)[pyg_graph_train.pseudo_val_mask])
            print("y: ", pyg_graph_train.y[pyg_graph_train.pseudo_val_mask])
        
        if retrain is True:
            model.reset_parameters()