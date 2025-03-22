import numpy as np
import pandas as pd
import os
import torch
from sklearn.manifold import TSNE
from src.feature_extraction.feature_extraction import clip_features
from src.utils.constants import EVENT_AUG_DIFF, TRAIN_SET 
from src.utils.utils import process_dataframe
import matplotlib.pyplot as plt

event = "mexico_earthquake"
device = "cpu"
dim = 2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

event_fantasy = event.replace("_", " ").title()

print(event_fantasy)

label_mapping = {0: 'not informative', 1: 'informative', 2: 'augmentation dataset'}
# Replace original labels with new labels

[df_text_train, _, _] = clip_features(mode="text", device=device, event_features=event)
[df_image_train, _, _] = clip_features(mode="image", device=device, event_features=event)
    
data_train = pd.read_json(TRAIN_SET.format(event), lines=True)
ft_train_images, ft_train_text, annot_train, _, _ = process_dataframe(data_train,
                                                                df_image_train,
                                                                df_text_train)

annot_train = np.column_stack([annot_train, np.zeros(annot_train.shape[0])])

aug_event = EVENT_AUG_DIFF[event]

for ix, ae in enumerate(aug_event):

    [df_text_aug, _, _] = clip_features(event_features=ae, mode="text", device=device)

    [df_image_aug, _, _] = clip_features(event_features=ae, mode="image", device=device)
    
    df_image_aug["labels"] = 2
    df_text_aug["labels"] = 2

    data_aug = pd.read_json(TRAIN_SET.format(ae), lines=True)
    ft_aug_images_e, ft_aug_text_e, annot_aug_e, _, _ = process_dataframe(data_aug, df_image_aug, df_text_aug)

    if ix == 0:
        ft_aug_images = ft_aug_images_e
        ft_aug_text = ft_aug_text_e
        annot_aug = annot_aug_e
    else:
        ft_aug_images = torch.concat([ft_aug_images, ft_aug_images_e])
        ft_aug_text = torch.concat([ft_aug_text, ft_aug_text_e])
        annot_aug = np.concatenate([annot_aug, annot_aug_e])

if annot_train.shape[0] < annot_aug.shape[0]:
    sample_size = annot_train.shape[0]
    indices = torch.randperm(annot_aug.shape[0])[:sample_size]
    
    ft_aug_images = ft_aug_images[indices]
    ft_aug_text = ft_aug_text[indices]
    annot_aug = annot_aug[indices]

annot_aug = np.zeros_like(annot_aug)
annot_aug = np.column_stack([annot_aug, np.ones(annot_aug.shape[0])])

ft_train_images = torch.concat([ft_train_images, ft_aug_images])
ft_train_text = torch.concat([ft_train_text, ft_aug_text])
annotations = np.concatenate([annot_train, annot_aug])

new_labels = np.array([label_mapping[np.argmax(label)] for label in annotations])

ft_mt_training_step = torch.concat([ft_train_images, ft_train_text], axis=1).cpu().numpy()

# Perform t-SNE embedding

if dim == 3:
    tsne = TSNE(n_components=3, random_state=13)
    embedded_data = tsne.fit_transform(ft_mt_training_step)
    
    # Create a figure for the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for annotations (assuming annotations are categorical)
    unique_annotations = np.unique(annotations, axis=0)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_annotations)))  # Generate colors based on number of unique annotations
    
    # Plot each point with its corresponding annotation color
    for i in range(len(unique_annotations)):
        indices = np.all(annotations == unique_annotations[i], axis=1)
        ax.scatter(embedded_data[indices, 0], embedded_data[indices, 1], embedded_data[indices, 2], c=[colors[i]], label=f'Annotation {i}')
    
    # Set labels and legend
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.title('t-SNE Visualization - {}'.format(event_fantasy))
    ax.legend()
    
    # Save the plot
    plt.savefig('tsne_visualization_3d.pdf')
    plt.close(fig)

elif dim == 2:
    # Perform t-SNE embedding
    tsne = TSNE(n_components=2, random_state=13)
    tsne_result = tsne.fit_transform(ft_mt_training_step)

    # Plotting
    plt.figure(figsize=(10, 8))
    for label in np.unique(new_labels):
        idx = new_labels == label
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=label, alpha=0.5)
    
    # Remove x-ticks and y-ticks
    plt.xticks([])
    plt.yticks([])

    # Set labels and legend
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization - {}'.format(event_fantasy), fontsize=20)
    plt.legend(fontsize=16)
    
    os.makedirs("plots/visualizations", exist_ok=True)

    # Save the plot as an image file
    plt.savefig('plots/visualizations/tsne_visualization_2d_{}.png'.format(event))
    plt.savefig('plots/visualizations/tsne_visualization_2d_{}.pdf'.format(event))

    # Optionally, you can also save as PDF or other formats
    # plt.savefig('tsne_visualization_2d.pdf')

    # Close the plot to free up memory
    plt.close()