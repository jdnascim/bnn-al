import os
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch
import tqdm

from src.arch.autoencoder import Autoencoder
from src.utils.constants import RED_CACHE


def pca_reduction(n_components, ft_labeled, ft_unlabeled, ft_dev, ft_aug_unlbl=None):

    if ft_aug_unlbl is None:
        full_ft_train = np.concatenate([ft_labeled, ft_unlabeled], axis=0)
    else:
        full_ft_train = np.concatenate([ft_labeled, ft_unlabeled, ft_aug_unlbl], axis=0)
    
    pca = PCA(n_components=n_components, random_state=13)
    pca.fit(full_ft_train)
    
    ft_labeled = pca.transform(ft_labeled)
    ft_unlabeled = pca.transform(ft_unlabeled)
    ft_dev = pca.transform(ft_dev)

    if ft_aug_unlbl is None:
        return ft_labeled, ft_unlabeled, ft_dev
    else:
        ft_aug_unlbl = pca.transform(ft_aug_unlbl)
        return ft_labeled, ft_unlabeled, ft_dev, ft_aug_unlbl
    

def autoencoder_reduction_old(arch_name, ft_labeled, ft_unlabeled, ft_dev, device, ft_aug_unlbl=None):
    
    if ft_aug_unlbl is None:
        full_ft = np.concatenate([ft_labeled, ft_unlabeled], axis=0)
    else:
        full_ft = np.concatenate([ft_labeled, ft_unlabeled, ft_aug_unlbl], axis=0)

    autoenc= Autoencoder(full_ft.shape[1], arch_name).to(device)

    optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    scaler = autoenc.scaler()
    full_ft_scaled = scaler.fit_transform(full_ft)
    full_ft_scaled = torch.Tensor(full_ft_scaled).to(device)

    num_epochs = autoenc.arch_info["epochs"]
    autoenc.train()
    with tqdm.trange(num_epochs, unit="epoch", mininterval=0, position=0, leave=True ) as bar:
        for epoch in range(num_epochs):
            bar.set_description(f"Epoch {epoch+1}")

                # Forward pass
            outputs = autoenc(full_ft_scaled)
            loss = criterion(outputs, full_ft_scaled)
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.update(1)
            bar.set_postfix(
               loss=float(loss)
            )
    
    autoenc.eval()
    enc = autoenc.encoder

    ft_labeled = scaler.transform(ft_labeled)
    ft_unlabeled = scaler.transform(ft_unlabeled)
    ft_dev = scaler.transform(ft_dev)

    ft_labeled = enc(torch.Tensor(ft_labeled).to(device)).detach().float()
    ft_unlabeled = enc(torch.Tensor(ft_unlabeled).to(device)).detach().float()
    ft_dev = enc(torch.Tensor(ft_dev).to(device)).detach().float()

    if ft_aug_unlbl is None:
        return ft_labeled, ft_unlabeled, ft_dev
    else:
        ft_aug_unlbl = enc(torch.Tensor(ft_aug_unlbl).to(device)).detach().float()
        return ft_labeled, ft_unlabeled, ft_dev, ft_aug_unlbl


def autoencoder_reduction(arch_name, full_ft, ft_dev, device, ft_source, event):
    filepath = RED_CACHE.format("autoenc", ft_source, event)

    if os.path.exists(filepath):
        with open(filepath, 'rb') as fp:
            vecs = pickle.load(fp)
        return vecs
    
    autoenc = Autoencoder(full_ft.shape[1], arch_name).to(device)

    optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    scaler = autoenc.scaler()
    full_ft_scaled = scaler.fit_transform(full_ft)
    full_ft_scaled = torch.Tensor(full_ft_scaled).to(device)

    num_epochs = autoenc.arch_info["epochs"]
    autoenc.train()
    with tqdm.trange(num_epochs, unit="epoch", mininterval=0, position=0, leave=True ) as bar:
        for epoch in range(num_epochs):
            bar.set_description(f"Epoch {epoch+1}")

                # Forward pass
            outputs = autoenc(full_ft_scaled)
            loss = criterion(outputs, full_ft_scaled)
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.update(1)
            bar.set_postfix(
               loss=float(loss)
            )
    
    autoenc.eval()
    enc = autoenc.encoder

    full_ft = scaler.transform(full_ft)
    ft_dev = scaler.transform(ft_dev)

    full_ft = enc(torch.Tensor(full_ft).to(device)).detach().float()
    ft_dev = enc(torch.Tensor(ft_dev).to(device)).detach().float()

    with open(filepath, "wb") as fp:
        full_ft.to("cpu")
        ft_dev.to("cpu")
        pickle.dump([full_ft, ft_dev], fp)     
        full_ft.to(device)
        ft_dev.to(device)

    return full_ft, ft_dev

