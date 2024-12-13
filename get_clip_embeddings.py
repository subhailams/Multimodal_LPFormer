# Code to run more data
import numpy as np
import os


import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from builder import PandasGraphBuilder
from data_utils import train_test_split_by_time, build_train_graph, build_val_test_matrix
# import dgl
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import pandas as pd

df = pd.read_csv('pinterest_iccv/board_pin_category_imgid_dataset_new.csv')
df.head(10)


df['board_id'] = pd.to_numeric(df['board_id'], errors='coerce')
df['pin_id'] = pd.to_numeric(df['pin_id'], errors='coerce')

# Drop rows where either 'board_id' or 'pin_id' is NaN (i.e., where they were non-numeric)
df = df.dropna(subset=['board_id', 'pin_id'])

# Reset the DataFrame index after dropping rows
df = df.reset_index(drop=True)


print("Number of Users/Boards: ",df['board_id'].unique().shape[0])
print("Number of Images/Pins: ",df['pin_id'].unique().shape[0])
print("Number of User-Item Interactions: ",df.shape[0])



directory = "pinterest_iccv"
out_directory = "./data_processed_clip_blip_100k"
image_directory = "/data/silama3/pinterest_images"
device = torch.device("cuda")
os.makedirs(out_directory, exist_ok=True)




# Directory where images are stored

# Filter the DataFrame to get only rows where category_name
india_df = df[df['category_name'] == 'Japan']

# Set a random seed for reproducibility
random.seed(125)

# Get unique img_ids from the filtered DataFrame
img_ids = india_df['img_id'].unique()

selected_img_ids = random.sample(list(img_ids), min(150, len(img_ids)))

# Load and store images
images = []
for img_id in selected_img_ids:
    image_path = os.path.join(image_directory, f"{img_id}.jpg")
    try:
        image = Image.open(image_path)
        images.append((img_id, image))
    except FileNotFoundError:
        # print(f"Image for img_id {img_id} not found.")
        pass

# Plot images in a 2x5 grid
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample images from Pinterest ICCV dataset", fontsize=16)

for ax, (img_id, image) in zip(axs.flatten(), images):
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"Img ID: {img_id}")

# Hide any unused subplots if there are fewer than 10 images
for ax in axs.flatten()[len(images):]:
    ax.axis("off")



if True:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    clip_embeddings = []
    blip_captions = []
    valid_img_ids = []

    for img_id in img_ids:
        image_path = os.path.join(image_directory, f"{img_id}.jpg")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")

                # Compute CLIP embedding
                clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    clip_embedding = clip_model.get_image_features(**clip_inputs)
                clip_embeddings.append(clip_embedding.cpu().numpy())

                # Generate BLIP caption
                blip_inputs = blip_processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = blip_model.generate(**blip_inputs)
                    blip_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
                blip_captions.append(blip_caption)

                valid_img_ids.append(img_id)

                # Stack CLIP embeddings
                clip_embeddings_stacked = np.vstack(clip_embeddings)

                # Convert valid_img_ids to numpy array
                valid_img_ids_array = np.array(valid_img_ids)

                # Convert blip_captions to numpy array of strings
                blip_captions_array = np.array(blip_captions, dtype=object)

                # Save all data to a single .npz file
                np.savez(os.path.join(out_directory,'image_data.npz'),
                        clip_embeddings=clip_embeddings_stacked,
                        blip_captions=blip_captions_array,
                        valid_img_ids=valid_img_ids_array)

                # print(f"Saved {len(valid_img_ids)} processed images.")
                # print(f"All data (CLIP embeddings, BLIP captions, and valid image IDs) saved to 'image_data.npz'")
            except Exception as e:
                # print(f"Error processing image {img_id}: {e}")
                continue
