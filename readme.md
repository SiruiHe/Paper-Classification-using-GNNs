#  Scientific Paper Classification using Graph Neural Networks 

This project implements various GNN architectures for paper classification on a custom ogbn-arxiv-like dataset.

## Project Structure

```
GNN_Project/
├── dataset/
│   ├── raw/                    # Raw data files
│   │   ├── edges.csv          # Paper citation network
│   │   ├── reduced_features_strict.csv  # SciBERT embeddings
│   │   ├── w2v_features_strict.csv      # Word2Vec embeddings
│   │   ├── node_labels.csv    # Paper category labels
│   │   └── paper_info.csv   # Paper information
│   └── create_dataset.py      # Dataset loader implementation
├── my_models.py               # GNN model implementations
└── train.py                   # Training pipeline
```

## Important: Data Preparation

Before running any experiments, make sure to:

1. Create the `dataset/raw/` directory if it doesn't exist
2. Place all required data files in the `dataset/raw/` directory:
   - `edges.csv`: Source and target node pairs for citations
   - `reduced_features_strict.csv`: SciBERT + PCA embeddings (128 dim)
   - `w2v_features_strict.csv`: Word2Vec embeddings (128 dim)
   - `node_labels.csv`: Paper category labels
   - `train_idx.npy`, `valid_idx.npy`, `test_idx.npy`: Data splits

## Models

Currently implemented models:

- MLP (Baseline)
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE

Each model supports:

- Different preprocessing methods ('basic' and 'arxiv-specific')
- Multiple embedding options (SciBERT and Word2Vec)
- Early stopping and model checkpointing

## Usage

### Training from Python

```python
from train import train_gnn_model

# Train GCN with SciBERT embeddings
results = train_gnn_model(
    model_type='gcn',           # Options: 'mlp', 'gcn', 'gat', 'gat_dgl', 'sage'
    preprocess_mode='compare',  # Options: 'compare', 'basic', 'arxiv'
    embedding_type='scibert'    # Options: 'scibert', 'word2vec'
)
```

### Training from Command Line

```bash
python train.py
```

## Dataset Format

The dataset loader (`CustomArxivDataset`) expects the following files in `dataset/raw/`:

- `edges.csv`: Source and target node pairs for citations
- `reduced_features_strict.csv`: SciBERT + PCA embeddings (128 dim)
- `w2v_features_strict.csv`: Word2Vec embeddings (128 dim)
- `node_labels.csv`: Paper category labels
- `train_idx.npy`, `valid_idx.npy`, `test_idx.npy`: Data splits

## Results

Training results are automatically saved to:

- `results/`: JSON files containing model configurations and performance metrics
- `checkpoints/`: Model checkpoints (best validation performance)

## Dependencies

- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- pandas
- numpy
- dgl (for gat_dgl)
