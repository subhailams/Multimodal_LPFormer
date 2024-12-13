import numpy as np
import pandas as pd
import os
import torch
import dgl
from sklearn.model_selection import train_test_split
import pickle
from builder import PandasGraphBuilder
from data_utils import train_test_split_by_time, build_train_graph, build_val_test_matrix



out_directory = "./data_processed_clip_blip_100k"

directory = "pinterest_iccv"

item_metadata = pd.read_csv(os.path.join(directory, "board_pin_category_imgid_dataset_new.csv"))

assert 'board_id' in item_metadata.columns and 'pin_id' in item_metadata.columns and 'category_name' in item_metadata.columns, \
        "board_pin_category_imgid_dataset.csv must contain 'board_id', 'pin_id', and 'category_name' columns."

item_metadata = item_metadata.rename(columns={'board_id': 'user_id', 'pin_id': 'item_id', 'img_id': 'img_id'})



# Load the data
data = np.load(os.path.join(out_directory, 'image_data.npz'), allow_pickle=True)

# Access the arrays
clip_embeddings = data['clip_embeddings']
blip_captions = data['blip_captions'].tolist()
valid_img_ids = data['valid_img_ids'].tolist()

print(f"Loaded {len(valid_img_ids)} processed images.")
print(f"CLIP embeddings shape: {clip_embeddings.shape}")
print(f"Number of BLIP captions: {len(blip_captions)}")
print(f"Number of valid image IDs: {len(valid_img_ids)}")

# Step 1: Filter item_metadata to include only rows with img_id in valid_img_ids
filtered_item_metadata = item_metadata[item_metadata['img_id'].isin(valid_img_ids)].drop_duplicates(subset=['img_id'])

print(f"Filtered item_metadata shape: {filtered_item_metadata.shape}")

# Step 2: Create interactions DataFrame based on the filtered metadata
interactions = filtered_item_metadata[['user_id', 'img_id', 'category_name']].copy()
interactions['timestamp'] = np.random.randint(0, 1000000, size=len(interactions))

# Step 3: Extract unique users and images
users = interactions[["user_id", "category_name"]].drop_duplicates()
images = filtered_item_metadata[["img_id"]].drop_duplicates()

# Convert to appropriate data types
users = users.astype({"user_id": "category", "category_name": "category"})
images = images.astype({"img_id": "category"})

print(f"Number of interactions: {interactions.shape[0]}")
print(f"Number of unique users: {users.shape[0]}")
print(f"Number of unique images: {images.shape[0]}")

# Map valid_img_ids to clip_embeddings
imgid_to_clip_embedding = {img_id: embedding for img_id, embedding in zip(valid_img_ids, clip_embeddings)}

print(f"Number of valid image IDs: {len(valid_img_ids)}")
print(f"Number of CLIP embeddings: {len(clip_embeddings)}")
print(f"Number of mapped img_id to embeddings: {len(imgid_to_clip_embedding)}")

# Prepare CLIP embeddings for graph assignment
clip_embeddings_for_graph = []
for img_id in images['img_id']:
    if img_id in imgid_to_clip_embedding:
        clip_embeddings_for_graph.append(imgid_to_clip_embedding[img_id])
    else:
        print(f"Warning: No embedding found for img_id {img_id}")

# Convert to numpy array for consistent shape
clip_embeddings_for_graph = np.vstack(clip_embeddings_for_graph)

print(f"Shape of CLIP embeddings for graph: {clip_embeddings_for_graph.shape}")
print(f"Number of images with CLIP embeddings: {images.shape[0]}")

# Initialize graph builder
graph_builder = PandasGraphBuilder()

# Add entities
graph_builder.add_entities(users, "user_id", "user")
graph_builder.add_entities(images, "img_id", "image")

# Add relationships
graph_builder.add_binary_relations(interactions, "user_id", "img_id", "interacts")
graph_builder.add_binary_relations(interactions, "img_id", "user_id", "interacted-by")

# Build graph
g = graph_builder.build()

# Assign user and image IDs as node features
g.nodes["user"].data["id"] = torch.LongTensor(users["user_id"].cat.codes.values)
g.nodes["image"].data["id"] = torch.LongTensor(images["img_id"].cat.codes.values)

# Assign clip_embeddings to image nodes
clip_embeddings_tensor = torch.FloatTensor(clip_embeddings_for_graph)
g.nodes["image"].data["clip_embedding"] = clip_embeddings_tensor

# Add timestamp as edge features
for edge_type in ["interacts", "interacted-by"]:
    g.edges[edge_type].data["timestamp"] = torch.LongTensor(interactions["timestamp"].values)

# Train-validation-test split
train_indices, val_indices, test_indices = train_test_split_by_time(
    interactions, "timestamp", "user_id"
)

edge_index_map = pd.Series(np.arange(len(interactions)), index=interactions.index)
train_indices = edge_index_map.loc[train_indices].values
val_indices = edge_index_map.loc[val_indices].values
test_indices = edge_index_map.loc[test_indices].values

# Build the graph with training interactions only
train_g = build_train_graph(
    g, train_indices, "user", "image", "interacts", "interacted-by"
)

# Ensure no disconnected nodes in the training graph
assert train_g.out_degrees(etype="interacts").min() > 0, "Some nodes have zero out-degree in training graph!"

# Build user-image sparse matrices for validation and test set
val_matrix, test_matrix = build_val_test_matrix(
    g, val_indices, test_indices, "user", "image", "interacts"
)

# Create image-category mapping
image_category_mapping = dict(zip(filtered_item_metadata['img_id'], filtered_item_metadata['category_name']))

# Create a list of category names corresponding to each image in the graph
image_categories = [
    image_category_mapping.get(img, "Unknown") for img in images["img_id"]
]

# Debug: Verify image categories
print(f"Sample image categories: {image_categories[:10]}")

# Save train graph
dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)

# Prepare dataset dictionary
dataset = {
    "val-matrix": val_matrix,
    "test-matrix": test_matrix,
    "user-type": "user",
    "item-type": "image",
    "user-to-item-type": "interacts",
    "item-to-user-type": "interacted-by",
    "timestamp-edge-column": "timestamp",
    "item-categories": image_categories,
}

# Save dataset to pickle file
with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
    pickle.dump(dataset, f)

print("Train graph and dataset saved successfully.")

# Additional debugging information
print(f"Number of nodes in train_g: {train_g.number_of_nodes()}")
print(f"Number of edges in train_g: {train_g.number_of_edges()}")
print(f"Node types in train_g: {train_g.ntypes}")
print(f"Edge types in train_g: {train_g.etypes}")