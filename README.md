
# Multimodal attention on LPFormer for Link Prediction on Pinterest Implementation

This repository contains the implementation of two variants: **LPFormer** and **Modified LPFormer with Multimodal Attention**, designed to handle multimodal data. The codebase includes modules for generating graphs, extracting CLIP embeddings, and implementing link transformers with multimodal modifications.

## File Descriptions

### 1. `train_model.py`
This script contains the main training loop for the LPFormer model. It handles data loading, model initialization, and training pipeline setup.

Key Features:
- Supports  LPFormer architectures.
- Implements multimodal data handling for text and vision.
- Includes hyperparameter tuning for optimal performance.

---

### 2. `link_transformer.py`
Implements the core functionality of the link transformer used in LPFormer. This file handles the linking between nodes in the graph.

Key Features:
- Standard link transformer implementation.
- Optimized for unimodal data processing.

---

### 3. `link_transformer_multimodal.py`
An extended version of the `link_transformer.py` designed for multimodal inputs. This version processes both visual and textual features to enhance link prediction accuracy.

Modifications:
- Added modules to handle and fuse embeddings from different modalities (e.g., text and vision).
- Modified loss functions to account for multimodal data.

---

### 4. `get_clip_embeddings.py`
A utility script to extract embeddings using the CLIP model for images and text.

Key Features:
- Extracts embeddings for text and visual inputs.
- Preprocesses data to ensure compatibility with the LPFormer pipeline.
- Outputs embeddings that can be used for graph generation or model input.

---

### 5. `generate_graph.py`
Generates graphs from the input data using CLIP embeddings or other features.

Key Features:
- Creates node and edge representations from embeddings.
- Supports multimodal data for graph generation.
- Includes options for dynamic or static graph construction.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up CLIP:
   - Ensure CLIP is installed and accessible. Use the official repository if necessary: [CLIP GitHub](https://github.com/openai/CLIP).

---

## Usage

### Generating Graphs
Run the `generate_graph.py` script to create graphs from embeddings:
```bash
python generate_graph.py --input_data <path-to-data> --output_graph <path-to-output>
```

### Extracting CLIP Embeddings
Use the `get_clip_embeddings.py` script:
```bash
python get_clip_embeddings.py --data <path-to-data> --output <path-to-embeddings>
```

### Training the Model
Train LPFormer or lpFormer using `train_model.py`:
```bash
python train_model.py --model lpformer --epochs 50 --learning_rate 0.001
```

### Multimodal Link Transformer
For multimodal data, use the `link_transformer_multimodal.py`:
```bash
python link_transformer_multimodal.py --input_graph <path-to-graph> --output_links <path-to-output>
```

---

## Multimodal Changes

- **Embedding Fusion**: Integrated CLIP embeddings for text and images.
- **Graph Construction**: Enhanced `generate_graph.py` to handle multimodal node and edge attributes.
- **Model Architecture**: Modified transformer layers in `link_transformer_multimodal.py` to incorporate attention mechanisms for different modalities.

---

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for embeddings.
- Research on LPFormer and lpFormer architectures for inspiration.

---


