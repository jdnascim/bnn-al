import argparse
from datetime import datetime
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from src.utils.gnn import run_base, generate_graph, validate_best_model
import torch
import pyro
import os
import pyro.distributions as dist
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO, Predictive
from src.feature_extraction.feature_extraction import maxvit_features, mpnet_features
from src.arch.bnn import BNN
from src.arch.base import BaseGNN, BaseMLP
import pandas as pd
import tqdm
from os.path import join
from tqdm.auto import trange
from src.utils.constants import DEV_SET, RESULT_FILE, SETUP_FILE, SPLIT_SET
from src.utils.utils import check_and_convert_to_tensor, process_dataframe, seed_everything
import yaml
from src.utils.reduction import autoencoder_reduction, pca_reduction


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
parser.add_argument("--autoenc", default="ae-base", type=str, required=False)
parser.add_argument("--pca_red", default=256, type=int, required=False)

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

df_image_train = df_image_train.set_index(['image_files','text'])
df_text_train = df_text_train.set_index(['image_files','text'])
df_image_dev = df_image_dev.set_index(['image_files','text'])
df_text_dev = df_text_dev.set_index(['image_files','text'])
df_image_test = df_image_test.set_index(['image_files','text'])
df_text_test = df_text_test.set_index(['image_files','text'])

df_image_train_dict = df_image_train.to_dict("index")
df_text_train_dict = df_text_train.to_dict("index")
df_image_dev_dict = df_image_dev.to_dict("index")
df_text_dev_dict = df_text_dev.to_dict("index")
df_image_test_dict = df_image_test.to_dict("index")
df_text_test_dict = df_text_test.to_dict("index")

# labeled
data_labeled = pd.read_json(SPLIT_SET.format(event, "labeled", labeled_size, set_id), lines=True)
ft_labeled_images, ft_labeled_text, annot_labeled = process_dataframe(data_labeled, df_image_train, df_text_train, df_image_train_dict, df_text_train_dict)

# unlabeled
data_unlbl = pd.read_json(SPLIT_SET.format(event, "unlabeled", labeled_size, set_id), lines=True)
ft_unlabeled_images, ft_unlabeled_text, annot_unlabeled = process_dataframe(data_unlbl, df_image_train, df_text_train, df_image_train_dict, df_text_train_dict)

# dev
data_dev = pd.read_json(DEV_SET.format(event), lines=True)
ft_dev_images, ft_dev_text, annot_dev = process_dataframe(data_dev, df_image_dev, df_text_dev, df_image_dev_dict, df_text_dev_dict)


if reduction == "pca":
    ft_labeled_images, ft_unlabeled_images, ft_dev_images = pca_reduction(args.pca_red,
                                                                        ft_labeled_images,
                                                                        ft_unlabeled_images,
                                                                        ft_dev_images)
    
    ft_labeled_text, ft_unlabeled_text, ft_dev_text = pca_reduction(args.pca_red,
                                                                ft_labeled_text,
                                                                ft_unlabeled_text,
                                                                ft_dev_text)
elif reduction == "autoenc":
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

ft_labeled_images = check_and_convert_to_tensor(ft_labeled_images)
ft_labeled_text = check_and_convert_to_tensor(ft_labeled_text)
annot_labeled = check_and_convert_to_tensor(annot_labeled)
ft_unlabeled_images = check_and_convert_to_tensor(ft_unlabeled_images)
ft_unlabeled_text = check_and_convert_to_tensor(ft_unlabeled_text)
annot_unlabeled = check_and_convert_to_tensor(annot_unlabeled)
ft_dev_images = check_and_convert_to_tensor(ft_dev_images)
ft_dev_text = check_and_convert_to_tensor(ft_dev_text)
annot_dev = check_and_convert_to_tensor(annot_dev)

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

image_ft_size = ft_labeled_images.shape[1]
text_ft_size = ft_labeled_text.shape[1]

if arch_setup_data["group"] == "bnn":
    # Define the model and guide
    model = BNN(input_dim, output_dim, hidden_dim, 1)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    
    # Set up the SVI optimizer and loss function
    optimizer = pyro.optim.Adam({"lr": 1e-4})
    elbo = Trace_ELBO()
    
    # Set up the SVI object
    svi = SVI(model, guide, optimizer, loss=elbo)
    
    # Train the model
    model.to(device)
    ft_labeled.to(device)
    
    num_iterations = 25000
    progress_bar = trange(num_iterations)
    
    for j in progress_bar:
        loss = svi.step(ft_labeled, annot_labeled)
        progress_bar.set_description("[iteration %04d] loss: %.4f" % (j + 1, loss / len(ft_labeled)))
        
    # Predict probabilities for test data
    predictive = Predictive(model, guide=guide, num_samples=500)
    
    ft_dev.to(device)
    y_pred = predictive(ft_dev)
    y_pred = y_pred["obs"].T.detach().numpy().mean(axis=1)
    
    bin_pred = (y_pred > 0.5).astype(int)
elif arch_setup_data["group"] == "mlp":
    model = BaseMLP(input_dim, output_dim, hidden_dim, n_hidden)
    model = model.to(device)
    ft_labeled = ft_labeled.to(device)
    annot_labeled = annot_labeled.to(device)

    print(model)
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

    model = BaseGNN(input_dim, output_dim, hidden_dim, n_hidden)
    model = model.to(device)

    print(model)

    pyg_graph_train = generate_graph(ft_mt_training_step, annot_mt_training_step, 16, labeled_ix=labeled_ix, unlabeled_ix=unlabeled_ix)
    pyg_graph_train = pyg_graph_train.to(device)
    
    best_model = run_base(model, pyg_graph_train, **kwargs)

    # dev-test graph
    ft_labeled = ft_labeled_copy
    ft_unlabeled = ft_unlabeled_copy
    ft_dev = ft_dev_copy

    ft_mt_dev_step = torch.concat([ft_labeled, ft_unlabeled, ft_dev])
    annot_mt_dev_step = torch.concat([annot_labeled, annot_unlabeled, annot_dev])

    labeled_ix = np.arange(0, ft_labeled.shape[0])
    unlabeled_ix = np.arange(ft_labeled.shape[0], ft_labeled.shape[0] + ft_unlabeled.shape[0])
    dev_ix = np.arange(ft_labeled.shape[0] + ft_unlabeled.shape[0], ft_mt_dev_step.shape[0])

    pyg_graph_dev = generate_graph(ft_mt_dev_step, annot_mt_dev_step, 16, labeled_ix=labeled_ix, unlabeled_ix=unlabeled_ix, test_ix=dev_ix)
    pyg_graph_dev = pyg_graph_dev.to(device)

    best_model.eval()

    resf = RESULT_FILE.format(event, exp_group, exp_id, labeled_size, set_id, run_id)
    os.makedirs(resf[:resf.rfind("/")], exist_ok=True)

    validate_best_model(best_model, pyg_graph_dev, resf)
