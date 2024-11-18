## Python environment setup with Conda

```bash
pip install pandas
pip install scikit_learn
pip install numpy
pip install scipy
pip install einops
pip install ogb torch-geometric
pip install pyyaml
pip install networkx
pip install gdown
pip install matplotlib
```

Training with 

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