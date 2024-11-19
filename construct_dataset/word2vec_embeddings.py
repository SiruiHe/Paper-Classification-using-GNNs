import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from tqdm import tqdm

nltk.download('punkt')

def train_word2vec(sorted_df):
    """在物理论文数据上训练Skip-gram模型"""
    print("准备语料库...")
    corpus = []
    
    # 准备训练语料：标题和摘要
    for _, row in tqdm(sorted_df.iterrows(), desc="处理文本"):
        text = f"{row['Title']} {row['Abstract']}"
        words = simple_preprocess(text)
        corpus.append(words)
    
    print("Training Skip-gram model...")
    model = Word2Vec(
        sentences=tqdm(corpus, desc="Training Word2Vec model"),
        vector_size=128,     # Same dimension as original implementation
        window=5,            # Context window size
        min_count=5,         # Minimum word frequency threshold
        workers=4,           # Number of parallel training threads
        sg=1                 # Use Skip-gram model (instead of CBOW)
    )
    
    return model.wv

def get_word_embeddings(text, word_vectors):
    """获取文本的词向量平均值"""
    words = simple_preprocess(text)
    word_vectors_list = []
    
    for word in words:
        if word in word_vectors:
            word_vectors_list.append(word_vectors[word])
    
    if not word_vectors_list:
        return np.zeros(128)  # 如果没有找到任何词，返回零向量
    
    return np.mean(word_vectors_list, axis=0)

def test_embeddings(sorted_df, word_vectors):
    """测试前三个样本的嵌入"""
    print("测试前三个节点的文本嵌入：")
    print("-" * 80)
    
    for idx, row in sorted_df.head(3).iterrows():
        print(f"Node {idx}:")
        print(f"Title: {row['Title']}")
        print(f"Abstract: {row['Abstract'][:200]}...")
        
        text = f"{row['Title']} {row['Abstract']}"
        embedding = get_word_embeddings(text, word_vectors)
        
        print(f"\n嵌入向量维度: {embedding.shape}")
        print(f"嵌入向量前5个值: {embedding[:5]}")
        print(f"嵌入向量L2范数: {np.linalg.norm(embedding):.4f}")
        print("-" * 80)

def main():
    # 读取数据
    print("加载论文数据...")
    sorted_df = pd.read_csv('raw/sorted_papers.csv')
    
    # 选择数据集后缀
    dataset_suffix = "_strict"  # 可选: "_strict", "_subseq", "_coarse"
    
    # 训练Word2Vec模型
    word_vectors = train_word2vec(sorted_df)
    print(f"词向量维度: {word_vectors.vector_size}")
    
    # 测试前三个样本
    print("\n=== 测试前三个样本 ===")
    test_embeddings(sorted_df, word_vectors)
    
    # 为所有论文生成特征
    features = []
    print("\n为所有论文生成嵌入...")
    
    for _, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
        text = f"{row['Title']} {row['Abstract']}"
        embedding = get_word_embeddings(text, word_vectors)
        features.append(embedding)
    
    # 转换为numpy数组并保存
    features = np.array(features)
    features_df = pd.DataFrame(features)
    features_df.columns = [f'feature{i+1}' for i in range(features.shape[1])]
    
    print(f"特征矩阵形状: {features.shape}")
    features_df.to_csv(f'raw/w2v_features{dataset_suffix}.csv', index=False)
    print(f"特征已保存到 w2v_features{dataset_suffix}.csv")

if __name__ == '__main__':
    main()