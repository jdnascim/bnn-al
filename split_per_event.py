import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.feature_extraction.feature_extraction import maxvit_features, mpnet_features

SPLIT_TYPE = "balanced_random"
np.random.seed(13)

labeled_sizes = [14]

# Write data to separate JSONL files
directory = "data/CrisisMMD_v2.0_baseline_split/data_splits/informative_orig"
for filename in os.listdir(directory):
    if filename.endswith('.jsonl'):
        filepath = os.path.join(directory, filename)

        # Read JSONL file into a DataFrame
        df = pd.read_json(filepath, lines=True, orient='records', encoding='utf-8')

        # Extract event from image path
        df['event'] = df['image'].str.split('/').str[1]

        # Create directory if not exists
        for event in df['event'].unique():
            event_dir = f"data/CrisisMMD_v2.0_baseline_split/event_data_splits/{event}"
            os.makedirs(event_dir, exist_ok=True)

        # Write DataFrame to separate JSONL files based on event
        for event, group_df in df.groupby('event'):
            event_filepath = f"data/CrisisMMD_v2.0_baseline_split/event_data_splits/{event}/{filename}"
            group_df.to_json(event_filepath, orient='records', lines=True, force_ascii=False)
        
            if filename == "train.jsonl":
                split_dir = f"data/CrisisMMD_v2.0_baseline_split/event_data_splits/{event}/al_splits/{SPLIT_TYPE}"
                os.makedirs(split_dir, exist_ok=True)

                dataset_size = len(group_df)

                for s in labeled_sizes:
                    if s < dataset_size:
                        for i in range(10):
                            if SPLIT_TYPE == "random":
                                labeled_ix = np.random.choice(np.arange(dataset_size), s, replace=False)
                                unlabeled_ix = np.array([j for j in range(dataset_size) if j not in labeled_ix])
                            elif SPLIT_TYPE == "balanced_random":
                                group_df['index_column'] = group_df.reset_index().index
                                grouped = group_df.groupby('label')

                                labeled_data = []
                                unlabeled_data = []

                                # Randomly select 5 samples from each label for the training set
                                for label, group in grouped:
                                    train_samples, test_samples = train_test_split(group, train_size=5, random_state=i)
                                    labeled_data.append(train_samples)
                                    unlabeled_data.append(test_samples)

                                # Combine train and test data into DataFrames
                                labeled_df = pd.concat(labeled_data)
                                unlabeled_df = pd.concat(unlabeled_data)  

                                labeled_ix = labeled_df["index_column"].to_numpy()
                                unlabeled_ix = unlabeled_df["index_column"].to_numpy()

                                group_df.drop('index_column', axis=1, inplace=True)
                            elif SPLIT_TYPE == "kmeans":
                                
                                [df_text_train, _, _] = mpnet_features('cuda:5', event)
                                [df_image_train, _, _] = maxvit_features('cuda:5', event)
                                a=1


                            group_df.iloc[labeled_ix].to_json(f"{split_dir}/labeled_s{s}_{i}.jsonl", orient='records', lines=True, force_ascii=False)

                            group_df.iloc[unlabeled_ix].to_json(f"{split_dir}/unlabeled_s{s}_{i}.jsonl", orient='records', lines=True, force_ascii=False)
            

