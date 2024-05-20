import argparse
import copy
from datetime import datetime

from sklearn.model_selection import train_test_split
from src.arch.utils import read_setup
from src.al.al import al_update
from src.gnn.utils import run_base, generate_graph, validate_best_model
import torch
import os
import torch.nn.functional as F
from src.arch.bnn import BayesianGNN, BayesianHybrid, BayesianMLP
from src.arch.base import BaseMLP
from tqdm.auto import trange
from src.utils.constants import DEV_SET, RESULT_FILE, SETUP_FILE, AL_SPLIT_SET, TRAIN_SET, WANDB_NAME
from src.utils.utils import seed_everything
import yaml
import numpy as np

import wandb
wandb.login()


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
parser.add_argument("--al_isel", default="random", type=str, required=False)
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
    kwargs.update({"aug_unlbl_set": True})
else:
    aug_unlbl_set = False
    kwargs.update({"aug_unlbl_set": False})

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

arch_setup_data = read_setup(arch_setup)

kwargs.update(arch_setup_data)

wandb.init(
    # Set the project where this run will be logged
    project="bnn-al", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name = WANDB_NAME.format(exp_id, event, labeled_size, set_id, run_id),
    # Track hyperparameters and run metadata
    config=kwargs
    )

kwargs.update({"wandb": wandb})

if dev_id is not None:
    device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

kwargs.update({"device": device})

use_gpu = device != 'cpu'

seed_everything(set_id)

if arch_setup_data["group"] == "mlp":
    model = BayesianMLP(**kwargs)
elif arch_setup_data["group"] == "gnn":
    model = BayesianGNN(**kwargs)
elif arch_setup_data["group"] == "hybrid":
    model = BayesianHybrid(**kwargs)

print(kwargs)
model = model.to(device)

pyg_graph_train, pyg_graph_dev = generate_graph(**kwargs)
pyg_graph_train = pyg_graph_train.to(device)
pyg_graph_dev = pyg_graph_dev.to(device)

for i in range(args.al_iter + 1):
    
    best_model = run_base(model, pyg_graph_train, **kwargs)

    resf = RESULT_FILE.format(event, exp_group, exp_id, labeled_size, set_id, run_id)
    os.makedirs(resf[:resf.rfind("/")], exist_ok=True)

    validate_best_model(best_model, pyg_graph_dev, resf, **kwargs)

    # Active Learning
    if args.al is not None and i < al_iter:
        print("AL Iteration: {} of {}".format(i+1, al_iter))
        model = best_model

        pyg_graph_train, pyg_graph_dev = al_update(model, pyg_graph_train, pyg_graph_dev, **kwargs)

        labeled_size += al_batch
            
    if retrain is True:
        model.reset_parameters()

# Mark the run as finished
wandb.finish()