import pandas as pd

# 读取处理后的 CSV 文件
processed_file = 'processed_large_file.csv'
df = pd.read_csv(processed_file)

# 删除 DOI 为空的行
df = df[df['DOI'].notna()]

# 将结果保存到一个新文件中
output_filtered_file = 'processed_with_doi.csv'
df.to_csv(output_filtered_file, index=False)

print("Filtered file saved with only rows that contain DOI.")
