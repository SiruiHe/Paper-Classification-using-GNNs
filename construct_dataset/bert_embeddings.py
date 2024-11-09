import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_scibert_embedding(text, tokenizer, model, max_length=512):
    """获取文本的SciBERT嵌入"""
    inputs = tokenizer(text, 
                      return_tensors="pt",
                      max_length=max_length,
                      truncation=True,
                      padding=True)
    
    # 将输入移到GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        avg_embeddings = summed / torch.sum(mask, dim=1)
        
    return avg_embeddings.cpu().numpy()[0]

def test_embeddings(sorted_df, tokenizer, model):
    """测试前三个样本的嵌入"""
    print("测试前三个节点的文本嵌入：")
    print("-" * 80)
    
    for idx, row in sorted_df.head(3).iterrows():
        print(f"Node {idx}:")
        print(f"DOI: {row['DOI']}")
        print(f"Title: {row['Title']}")
        print(f"Abstract: {row['Abstract'][:200]}...")  # 只显示摘要的前200个字符
        
        # 拼接文本并获取嵌入
        text = f"{row['Title']} [SEP] {row['Abstract']}"
        embedding = get_scibert_embedding(text, tokenizer, model)
        
        print(f"\n嵌入向量维度: {embedding.shape}")
        print(f"嵌入向量前5个值: {embedding[:5]}")
        print(f"嵌入向量L2范数: {np.linalg.norm(embedding):.4f}")
        print("-" * 80)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和分词器
    print("Loading SciBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model = model.to(device)
    model.eval()
    
    # 读取数据
    print("Loading data...")
    sorted_df = pd.read_csv('sorted_papers.csv')
    
    # 测试前三个样本
    print("\n=== Testing first three samples ===")
    test_embeddings(sorted_df, tokenizer, model)
    
    # 询问是否继续处理所有数据
    response = input("\n测试结果看起来正确吗？是否继续处理所有数据？(y/n): ")
    if response.lower() != 'y':
        print("已终止处理")
        return
    
    # 创建特征矩阵
    features = []
    print("\nGenerating embeddings for all nodes...")
    
    for _, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
        text = f"{row['Title']} [SEP] {row['Abstract']}"
        embedding = get_scibert_embedding(text, tokenizer, model)
        features.append(embedding)
    
    # 转换为numpy数组并保存
    features = np.array(features)
    features_df = pd.DataFrame(features)
    features_df.columns = [f'feature{i+1}' for i in range(features.shape[1])]
    
    print(f"Feature matrix shape: {features.shape}")
    features_df.to_csv('node_features.csv', index=False)
    print("Features saved to node_features.csv")

if __name__ == '__main__':
    main()