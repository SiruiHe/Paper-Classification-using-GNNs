# Graph Machine Learning Project (CS 5284)

This project implements a custom graph dataset similar to ogbn-arxiv, along with GNN models for node classification.

> **Current version of data has some issues. Need to reconstruct the dataset using the new crawler data!**


## Setup Instructions
1. Place your `node_features.csv` file in `construct_dataset/raw/` directory
2. Run dataset construction
    ```bash
    python construct_dataset/create_dataset.py
    ```

## Training
Currently implemented models:
- GCN (Baseline)

To train the model:

```bash
python train.py
```


## TODO
- Implement additional GNN architectures
- Add model evaluation metrics
- Add visualization and more detailed EDA

## Dataset Format
- `edges.csv`: Source and target node pairs
- `node_features.csv`: Node feature vectors
- `node_labels.csv`: Node classification labels

## Notes
This project is part of CS 5284 Graph Machine Learning course assignment. The dataset structure follows the format of ogbn-arxiv but with our custom data.