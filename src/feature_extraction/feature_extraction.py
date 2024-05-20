from PIL import ImageFile, Image
import tqdm
import numpy as np
from src.feature_extraction.preprocess import pre_process
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from torch.utils.data import DataLoader
import torch
import timm
from os.path import join
from torch.utils.data import Dataset
from typing import List, Union, Tuple
import torchvision
from pathlib import PurePath
from src.feature_extraction.img_utils import load_image, check_ext
from src.feature_extraction.vector_utils import normalize_vector
import pickle
import os
from src.utils.constants import DATAPATH, EVENT_SPLIT_JSON, EVENTS, IMAGEPATH
from transformers import AutoProcessor, CLIPModel, AutoTokenizer


class ImageDataset(Dataset):
   
    def __init__(self,
                 image_files : Union[PurePath, str],
                 image_ids : List[int],
                 transform: torchvision.transforms,
                 hf_ret_tensors=False,
    ):
        
        # Remoe invalid image_files
        entry =  [(img_file, img_id) for img_file, img_id in 
                                     zip(image_files, image_ids)
                                     if check_ext(img_file)]
        
        self.image_files = [e[0] for e in entry ]
        self.image_ids = [e[1] for e in entry ]
        self.transform = transform
        self.hf_ret_tensors = hf_ret_tensors

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image = load_image(img_file)
        img_id = self.image_ids[idx]
        if self.transform and self.hf_ret_tensors is False:
            image = self.transform(image)
        elif self.transform and self.hf_ret_tensors is True:
            image = self.transform(images=image, return_tensors='pt')
        return image, img_id


def mpnet_features(event_features=None, **kwargs):
    dev = kwargs.get("device")

    if event_features is None:
        event = kwargs.get("event")
    else:
        event = event_features

    name = 'all-mpnet-base-v2'

    filepath = "data/.cache/{}_{}.pkl".format(name, event)

    if os.path.exists(filepath):
        with open(filepath, 'rb') as fp:
            dfs = pickle.load(fp)
        return dfs

    dfs = [None, None, None]
    
    for ix, split in enumerate(("train", "dev", "test")):
        data = pd.read_json(EVENT_SPLIT_JSON.format(event, split), lines=True)

        labels = []
        clean_text = [] 
        text = []
        image_files = []
    
        for _, row in tqdm.tqdm(data.iterrows()):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            clean_text.append(pre_process(row['text'], keep_hashtag = True, keep_special_symbols = True))
            labels.append(row['label']) 
    
        # Get pretrained model
        model = SentenceTransformer(name, cache_folder='.')
        model.to(dev)
        model = model.eval()
    
        # Enccode text
        tweets_emds = model.encode(clean_text, device=dev)

    
        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                        "text": text,
                        "clean_text": clean_text,
                        "embeddings": [f for f in tweets_emds],
                        "labels": np.argmax(labels, axis=1)})
                    
        dfs[ix] = df
    
    with open(filepath, "wb") as fp:
        pickle.dump(dfs, fp)     

    return dfs
    

def maxvit_features(event_features=None, **kwargs):
    dev = kwargs.get("device")

    if event_features is None:
        event = kwargs.get("event")
    else:
        event = event_features

    dfs = [None, None, None]
    name = 'maxvit_tiny_tf_224.in1k'

    filepath = "data/.cache/{}_{}.pkl".format(name, event)

    if os.path.exists(filepath):
        with open(filepath, 'rb') as fp:
            dfs = pickle.load(fp)
        return dfs

    #embeddings, embeddings_ids  = get_image_embedding(image_files, image_ids, model, m_transform, normalize=True, gpu_id=0, use_gpu=True)
    model = timm.create_model(
        #'maxvit_xlarge_tf_512.in21k_ft_in1k',
        name,
        pretrained=True,
        num_classes=0, # remove classifier nn.Linear
    )
    model = model.to(dev)
    model = model.eval()
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms_model = timm.data.create_transform(**data_config, is_training=False)
    batch_size = 128
    

    for ix, split in enumerate(("train", "dev", "test")):
        embedded_vectors = []
        embeddings_ids = []
    
        data = pd.read_json(EVENT_SPLIT_JSON.format(event, split), lines=True)

        image_files = []
        text = []
        labels = []
        
        for _, row in tqdm.tqdm(data.iterrows()):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            # Applie Preprocess stage here
            labels.append(row['label']) 
        
        image_ids = [i for i in range(len(image_files))]
        
        img_dataset = ImageDataset(image_files, image_ids, transforms_model)
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
    
        for imgs, ids_batch in tqdm.tqdm(img_dataloader, total=np.ceil(len(img_dataset) / batch_size).astype(int)):
            imgs = imgs.to(dev)
            
            with torch.no_grad():
                output = model(imgs)  
                output = output.squeeze().cpu().numpy()
                
            output = normalize_vector(output)
                
            embedded_vectors += output.tolist()
            embeddings_ids += [int(_id) for _id in ids_batch]
        
        
        image_files = [image_files[i] for i in embeddings_ids]
        embeddings = [embedded_vectors[i] for i in embeddings_ids]
        labels = [labels[i] for i in embeddings_ids]

        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                            "text": text,
                            "embeddings": [f for f in embeddings],
                            "labels": np.argmax(labels, axis=1)})
    
        dfs[ix] = df

    
    with open(filepath, "wb") as fp:
        pickle.dump(dfs, fp)     

    return dfs


def clip_features(mode="image", event_features=None, **kwargs):
    dev = kwargs.get("device")

    if event_features is None:
        event = kwargs.get("event")
    else:
        event = event_features

    assert mode == "image" or mode == "text", "mode should be image or text"

    dfs = [None, None, None]

    filepath = "data/.cache/clip_{}_{}.pkl".format(mode, event)

    if os.path.exists(filepath):
        with open(filepath, 'rb') as fp:
            dfs = pickle.load(fp)
        return dfs

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(dev)
    if mode == "image":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        batch_size = 128
    elif mode == "text":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    

    for ix, split in enumerate(("train", "dev", "test")):
        embedded_vectors = []
        embeddings_ids = []
    
        data = pd.read_json(EVENT_SPLIT_JSON.format(event, split), lines=True)

        image_files = []
        text = []
        labels = []
        clean_text = []
        
        for _, row in tqdm.tqdm(data.iterrows()):
            image_files.append(join(IMAGEPATH, row['image']))
            text.append(row['text'])
            
            if mode == "text":
                clean_text.append(pre_process(row['text'], keep_hashtag = True, keep_special_symbols = True))
            # Applie Preprocess stage here
            labels.append(row['label']) 
        
        image_ids = [i for i in range(len(image_files))]

        full_ft = torch.zeros([len(image_files), 512])
        batch_ix = 0
        
        if mode == "image":
            img_dataset = ImageDataset(image_files, image_ids, processor, hf_ret_tensors=True)
            img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        
            for imgs, ids_batch in tqdm.tqdm(img_dataloader, total=np.ceil(len(img_dataset) / batch_size).astype(int)):
                imgs = imgs.to(dev)

                with torch.no_grad():
                    imgs.data["pixel_values"] = imgs.data["pixel_values"].squeeze()
                    image_features = model.get_image_features(**imgs)
                    
                # Append features and image IDs
                ini = batch_size * batch_ix
                end = min(batch_size * (batch_ix+1), len(image_files))
                full_ft[ini:end] = image_features
                embeddings_ids.extend(ids_batch)
                batch_ix += 1
            
            embedded_vectors = full_ft.detach().cpu().numpy()
        
            image_files = [image_files[i] for i in embeddings_ids]
            embeddings = [embedded_vectors[i] for i in embeddings_ids]
            labels = [labels[i] for i in embeddings_ids]

        elif mode == "text":
            inputs = tokenizer(clean_text, return_tensors="pt", padding="max_length", 
                               max_length=tokenizer.model_max_length, truncation=True)
            inputs.to(dev)
            
            with torch.no_grad():
                embeddings = model.get_text_features(**inputs)  
                
            embeddings = embeddings.detach().cpu().numpy()

        labels = [ (1,0) if l == 'not_informative' else (0,1) for l in labels ]
        labels = np.array(labels)
    
        df = pd.DataFrame({"image_files": image_files,
                            "text": text,
                            "embeddings": [f for f in embeddings],
                            "labels": np.argmax(labels, axis=1)})
    
        dfs[ix] = df

    
    with open(filepath, "wb") as fp:
        pickle.dump(dfs, fp)     

    return dfs


def aug_unlbl_features(gpu, event):
    df_train_image = None
    df_train_text = None
    for e in EVENTS:
        if e != event:
            [df_train_image_event, _, _] = maxvit_features(gpu, e)
            [df_train_text_event, _, _] = mpnet_features(gpu, e)

            # Add the 'event' column to both dataframes
            df_train_image_event.loc[:, 'event'] = e
            df_train_text_event.loc[:, 'event'] = e

            if df_train_image is None:
                df_train_image = df_train_image_event
            else:
                df_train_image = pd.concat([df_train_image, df_train_image_event])    

            if df_train_text is None:
                df_train_text = df_train_text_event
            else:
                df_train_text = pd.concat([df_train_text, df_train_text_event])    
        
    # Select necessary columns from df_train_image
    df_train_image_selected = df_train_image[['image_files', 'text', 'labels', 'event']]

    # Reorder columns
    data_aug_unlbl = df_train_image_selected[['labels', 'image_files', 'text', 'event']]

    data_aug_unlbl = data_aug_unlbl.rename(columns={'image_files': 'image'})
    data_aug_unlbl = data_aug_unlbl.rename(columns={'labels': 'label'})

    data_aug_unlbl['image'] = data_aug_unlbl['image'].apply(lambda x: x[len("./data/CrisisMMD_v2.0/"):] if x.startswith("./data/CrisisMMD_v2.0/") else x)

        
    return data_aug_unlbl, df_train_image, df_train_text