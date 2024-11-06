import pandas as pd

# 设置分块大小
chunksize = 10000  # 根据内存情况调整

# 设置输出文件路径
output_file = 'processed_large_file.csv'

# 处理文件时是否需要写入表头
write_header = True

# 按块读取和处理 CSV 文件
for i, chunk in enumerate(pd.read_csv('广义物理类，不含交叉.csv', chunksize=chunksize)):
    # 提取主要子类
    chunk['Primary_Subclass'] = chunk['Categories'].apply(lambda x: x.split(',')[0])


    # 提取 DOI
    def extract_doi(categories):
        parts = categories.split(',')
        for i, part in enumerate(parts):
            if part.strip() == 'doi' and i + 1 < len(parts):
                return parts[i + 1].strip()
        return None


    chunk['DOI'] = chunk['Categories'].apply(extract_doi)

    # 清理 Abstract 中的“△ Less”
    chunk['Abstract'] = chunk['Abstract'].str.replace(r'\s*△ Less$', '', regex=True)

    # 将处理后的块追加保存到 CSV 文件
    chunk.to_csv(output_file, mode='a', index=False, header=write_header)

    # 在第一次写入后，不再写入表头
    write_header = False

    # 输出进度
    print(f"Chunk {i + 1} processed and saved to CSV.")

print("All chunks processed and saved.")

