import os
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, balanced_accuracy_score
from os.path import join
from sklearn.model_selection import train_test_split
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



def custom_serializer(obj):
    if isinstance(obj, (list, tuple)):
        return ' '.join(str(x) for x in obj)
    else:
        return obj