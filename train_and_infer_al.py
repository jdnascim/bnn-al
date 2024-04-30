import argparse
from datetime import datetime
from src.al.al import bald_base, bald_kmeans, batchbald_sel, rdn_sel, unc_kmeans_sel, unc_sel
from src.utils.gnn import run_base, generate_graph, validate_best_model
import torch
import os
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO, Predictive
from src.feature_extraction.feature_extraction import aug_unlbl_features, maxvit_features, mpnet_features
from src.arch.bnn import BayesianGNN
from src.arch.base import BaseGNN, BaseMLP
import pandas as pd
from tqdm.auto import trange
from src.utils.constants import DEV_SET, RESULT_FILE, SETUP_FILE, AL_SPLIT_SET
from src.utils.utils import check_and_convert_to_tensor, process_dataframe, seed_everything
import yaml
from src.utils.reduction import autoencoder_reduction, pca_reduction
import numpy as np


start_time = datetime.now()

seed_everything(13)

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
parser.add_argument("--al", default=None, type=str, required=False, choices=["random", "unc", "kmeans", "unc-kmeans", "bald-base", "bald-kmeans", "batchbald"])
parser.add_argument("--al_iter", default=3, type=int)
parser.add_argument("--al_batch", default=10, type=int)
parser.add_argument("--al_isel", default="random", type=str, required=False, choices=["random", "balanced_random"])
parser.add_argument("--aug_unlbl_set", action="store_true", default=False)

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

if dev_id is not None:
    device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

use_gpu = device != 'cpu'

[df_text_train, df_text_dev, df_text_test] = mpnet_features(dev_id, event)
[df_image_train, df_image_dev, df_image_test] = maxvit_features(dev_id, event)


# labeled
print(AL_SPLIT_SET.format(event, al_isel, "labeled", labeled_size, set_id))
data_labeled = pd.read_json(AL_SPLIT_SET.format(event, al_isel, "labeled", labeled_size, set_id), lines=True)
ft_labeled_images, ft_labeled_text, annot_labeled = process_dataframe(data_labeled, df_image_train, df_text_train)

# unlabeled
data_unlbl = pd.read_json(AL_SPLIT_SET.format(event, al_isel, "unlabeled", labeled_size, set_id), lines=True)
ft_unlabeled_images, ft_unlabeled_text, annot_unlabeled = process_dataframe(data_unlbl, df_image_train, df_text_train)

# aug unlbl
if aug_unlbl_set is True:
    [data_aug_unlbl, df_aug_unlbl_image, df_aug_unlbl_text] = aug_unlbl_features(dev_id, event)
    ft_aug_unlbl_images, ft_aug_unlbl_text, annot_aug_unlbl = process_dataframe(data_aug_unlbl, df_aug_unlbl_image, df_aug_unlbl_text)

# dev
data_dev = pd.read_json(DEV_SET.format(event), lines=True)
ft_dev_images, ft_dev_text, annot_dev = process_dataframe(data_dev, df_image_dev, df_text_dev)


if reduction == "pca":
    if aug_unlbl_set is False:
        ft_labeled_images, ft_unlabeled_images, ft_dev_images = pca_reduction(args.pca_red,
                                                                            ft_labeled_images,
                                                                            ft_unlabeled_images,
                                                                            ft_dev_images)
        
        ft_labeled_text, ft_unlabeled_text, ft_dev_text = pca_reduction(args.pca_red,
                                                                    ft_labeled_text,
                                                                    ft_unlabeled_text,
                                                                    ft_dev_text)
    else:
        ft_labeled_images, ft_unlabeled_images, ft_dev_images, ft_aug_unlbl_images = pca_reduction(args.pca_red,
                                                                                                ft_labeled_images,
                                                                                                ft_unlabeled_images,
                                                                                                ft_dev_images,
                                                                                                ft_aug_unlbl_images)
        
        ft_labeled_text, ft_unlabeled_text, ft_dev_text, ft_aug_unlbl_text = pca_reduction(args.pca_red,
                                                                                            ft_labeled_text,
                                                                                            ft_unlabeled_text,
                                                                                            ft_dev_text,
                                                                                            ft_aug_unlbl_text)
elif reduction == "autoenc":
    if aug_unlbl_set is False:
        ft_labeled_images, ft_unlabeled_images, ft_dev_images = autoencoder_reduction(args.autoenc,
                                                                            ft_labeled_images,
                                                                            ft_unlabeled_images,
                                                                            ft_dev_images,
                                                                            device)
        
        ft_labeled_text, ft_unlabeled_text, ft_dev_text = autoencoder_reduction(args.autoenc,
                                                                    ft_labeled_text,
                                                                    ft_unlabeled_text,
                                                                    ft_dev_text,
                                                                    device)
    else:
        ft_labeled_images, ft_unlabeled_images, ft_dev_images, ft_aug_unlbl_images = autoencoder_reduction(args.autoenc,
                                                                            ft_labeled_images,
                                                                            ft_unlabeled_images,
                                                                            ft_dev_images,
                                                                            device,
                                                                            ft_aug_unlbl_images)
        
        ft_labeled_text, ft_unlabeled_text, ft_dev_text, ft_aug_unlbl_text = autoencoder_reduction(args.autoenc,
                                                                    ft_labeled_text,
                                                                    ft_unlabeled_text,
                                                                    ft_dev_text,
                                                                    device,
                                                                    ft_aug_unlbl_text)
        

ft_labeled_images = check_and_convert_to_tensor(ft_labeled_images)
ft_labeled_text = check_and_convert_to_tensor(ft_labeled_text)
annot_labeled = check_and_convert_to_tensor(annot_labeled)
ft_unlabeled_images = check_and_convert_to_tensor(ft_unlabeled_images)
ft_unlabeled_text = check_and_convert_to_tensor(ft_unlabeled_text)
annot_unlabeled = check_and_convert_to_tensor(annot_unlabeled)
ft_dev_images = check_and_convert_to_tensor(ft_dev_images)
ft_dev_text = check_and_convert_to_tensor(ft_dev_text)
annot_dev = check_and_convert_to_tensor(annot_dev)

event_unlbl_size = ft_unlabeled_images.shape[0]
if aug_unlbl_set is True:
    ft_aug_unlbl_images = check_and_convert_to_tensor(ft_aug_unlbl_images)
    ft_aug_unlbl_text = check_and_convert_to_tensor(ft_aug_unlbl_text)
    annot_aug_unlbl = check_and_convert_to_tensor(annot_aug_unlbl)

    ft_unlabeled_images = torch.concat([ft_unlabeled_images, ft_aug_unlbl_images], dim=0)
    ft_unlabeled_text = torch.concat([ft_unlabeled_text, ft_aug_unlbl_text], dim=0)
    annot_unlabeled = torch.concat([annot_unlabeled, annot_aug_unlbl], dim=0)

ft_labeled = torch.concat([ft_labeled_images, ft_labeled_text], dim=1)
ft_unlabeled = torch.concat([ft_unlabeled_images, ft_unlabeled_text], dim=1)
ft_dev = torch.concat([ft_dev_images, ft_dev_text], dim=1)

ft_labeled_copy = ft_labeled.clone()
ft_unlabeled_copy = ft_unlabeled.clone()
ft_dev_copy = ft_dev.clone()

annot_labeled = torch.argmax(annot_labeled, dim=1)
annot_unlabeled = torch.argmax(annot_unlabeled, dim=1)
annot_dev = torch.argmax(annot_dev, dim=1)

ft_mt_training_step = torch.concat([ft_labeled, ft_unlabeled])
annot_mt_training_step = torch.concat([annot_labeled, annot_unlabeled])

labeled_ix = np.arange(0, ft_labeled.shape[0])
unlabeled_ix = np.arange(ft_labeled.shape[0], ft_mt_training_step.shape[0])
event_unlbl_ix = np.arange(ft_labeled.shape[0], ft_labeled.shape[0] + event_unlbl_size)

image_ft_size = ft_labeled_images.shape[1]
text_ft_size = ft_labeled_text.shape[1]

if arch_setup_data["group"] == "mlp":
    model = BaseMLP(input_dim, output_dim, hidden_dim, n_hidden)
    model = model.to(device)
    ft_labeled = ft_labeled.to(device)
    annot_labeled = annot_labeled.to(device)

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
    ft_dev = ft_dev.to(device)
    y_pred = model(ft_dev)
    bin_pred = np.argmax(y_pred.detach().cpu().numpy(),axis=1)
elif arch_setup_data["group"] == "gnn":

    model = BayesianGNN()
    model = model.to(device)

    pyg_graph_train = generate_graph(ft_mt_training_step, annot_mt_training_step, 16, labeled_ix=labeled_ix,
                                     unlabeled_ix=unlabeled_ix, event_unlbl_ix=event_unlbl_ix)
    pyg_graph_train = pyg_graph_train.to(device)

    ft_mt_dev_step = torch.concat([ft_labeled, ft_unlabeled, ft_dev])
    annot_mt_dev_step = torch.concat([annot_labeled, annot_unlabeled, annot_dev])

    labeled_ix = np.arange(0, ft_labeled.shape[0])
    unlabeled_ix = np.arange(ft_labeled.shape[0], ft_labeled.shape[0] + ft_unlabeled.shape[0])
    event_unlbl_ix = np.arange(ft_labeled.shape[0], ft_labeled.shape[0] + event_unlbl_size)
    dev_ix = np.arange(ft_labeled.shape[0] + ft_unlabeled.shape[0], ft_mt_dev_step.shape[0])

    pyg_graph_dev = generate_graph(ft_mt_dev_step, annot_mt_dev_step, 16, labeled_ix=labeled_ix, 
                                   unlabeled_ix=unlabeled_ix, test_ix=dev_ix, event_unlbl_ix=event_unlbl_ix)
    pyg_graph_dev = pyg_graph_dev.to(device)

    for i in range(args.al_iter + 1):
        
        best_model = run_base(model, pyg_graph_train, **kwargs)

        # dev-test graph
        ft_labeled = ft_labeled_copy
        ft_unlabeled = ft_unlabeled_copy
        ft_dev = ft_dev_copy

        resf = RESULT_FILE.format(event, exp_group, exp_id, labeled_size, set_id, run_id)
        os.makedirs(resf[:resf.rfind("/")], exist_ok=True)

        validate_best_model(best_model, pyg_graph_dev, resf, **kwargs)

        # Active Learning
        if args.al is not None and i < al_iter:
            print("AL Iteration: {} of {}".format(i+1, al_iter))
            model = best_model
            if al == "random":
                selected_indices = rdn_sel(pyg_graph_train, al_batch)
                
            elif al == "unc":
                selected_indices = unc_sel(model, pyg_graph_train, al_batch)
              
            elif al == "unc-kmeans":
                selected_indices = unc_kmeans_sel(model, pyg_graph_train, al_batch)

            elif al == "bald-base":
                selected_indices = bald_base(model, pyg_graph_train, al_batch, bald_iter=10)
                
            elif al == "bald-kmeans":
                selected_indices = bald_kmeans(model, pyg_graph_train, al_batch, bald_iter=10)
            
            elif al == "batchbald":
                selected_indices = batchbald_sel(model, pyg_graph_train, al_batch, device, bald_iter=10)
                selected_indices = selected_indices.squeeze()
                
            labeled_size += len(selected_indices)
            pyg_graph_train.labeled_mask[selected_indices] = True
            pyg_graph_train.unlbl_mask[selected_indices] = False
            pyg_graph_dev.labeled_mask[selected_indices] = True
            pyg_graph_dev.unlbl_mask[selected_indices] = False