## Python environment setup with Conda

```bash
pip install pandas==2.2.2
pip install scikit_learn==1.5.2
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install einops==0.8.0
pip install ogb==1.3.6
pip install pyyaml==6.0.2
pip install networkx==3.4.2
pip install gdown==5.2.0
pip install matplotlib==3.8.0
pip install torch-geometric
```

## Datasets

These files should be placed in the directory "./data/"::
- edges.csv
- node_features_w2v_strict.csv
- node_labels.csv
- reduced_features_strict.csv
- test_idx.npy
- train_idx.npy
- valid_idx.npy


## Training

Please refer to run_models.ipynb

```bash
# GCN + scibert&pca
python main-arxiv.py --dataset arxiv-physics --hidden_channels 512 --epochs 500 --lr 0.0005 --runs 2 --local_layers 5 --embeddings scibert --bn --device 0 --res 
# GCN + word2vec&skip-gram
python main-arxiv.py --dataset arxiv-physics --hidden_channels 512 --epochs 500 --lr 0.0005 --runs 2 --local_layers 5 --embeddings word2vec --bn --device 0 --res 

# SAGE + scibert&pca
python main-arxiv.py --dataset arxiv-physics --hidden_channels 256 --epochs 500 --lr 0.0005 --runs 2 --local_layers 4 --embeddings scibert --bn --device 0 --res --sage
# SAGE + word2vec&skip-gram
python main-arxiv.py --dataset arxiv-physics --hidden_channels 256 --epochs 500 --lr 0.0005 --runs 2 --local_layers 4 --embeddings word2vec --bn --device 0 --res --sage

# GAT + scibert&pca
python main-arxiv.py --dataset arxiv-physics --hidden_channels 256 --epochs 500 --lr 0.0005 --runs 2 --local_layers 4 --embeddings scibert --bn --device 0 --res --local_attn
# GAT + word2vec&skip-gram
python main-arxiv.py --dataset arxiv-physics --hidden_channels 256 --epochs 500 --lr 0.0005 --runs 2 --local_layers 4 --embeddings word2vec --bn --device 0 --res --local_attn
```
