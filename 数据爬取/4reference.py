import pandas as pd
import requests
import csv
import time

# 读取已处理的 CSV 文件
input_file = 'processed_with_doi.csv'
df = pd.read_csv(input_file)

# 输出文件路径
output_file = '新版processed_with_references.csv'


def escape_csv_field(value):
    """对字段进行转义，确保字段中的逗号、换行符和引号不会导致CSV格式出错。"""
    if isinstance(value, str):
        # 去除换行符和空格
        value = value.replace('\n', ' ').replace('\r', '')
        # 如果包含引号，使用双引号进行转义
        if '"' in value:
            value = '"' + value.replace('"', '""') + '"'
    return value


# 定义一个函数，通过 CrossRef API 获取引用的 DOI
def get_references_doi(doi):
    if pd.isna(doi):
        return "No References"  # 如果 DOI 为 None，返回 "No References"

    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 提取引用文献的 DOI 列表
        references = data.get("message", {}).get("reference", [])
        reference_dois = [ref.get("DOI") for ref in references if "DOI" in ref]

        # 如果没有引用文献的 DOI，返回 "No References"
        if not reference_dois:
            return "No References"

        # 将引用的 DOI 以双竖线分隔的字符串返回
        return '||'.join(reference_dois)
    except requests.RequestException as e:
        print(f"Error fetching references for DOI {doi}: {e}")
        return "ERROR"  # 在发生错误时返回 "ERROR"


# 创建一个新的 CSV 文件并写入表头
with open(output_file, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)

    # 写入列名（表头）
    writer.writerow(['DOI', 'References_DOI'])

    # 处理每一行数据并保存
    for i, row in df.iterrows():
        doi = escape_csv_field(row['DOI'])  # 获取并转义 DOI 字段
        references = escape_csv_field(get_references_doi(doi))  # 获取并转义 References_DOI 字段

        # 将 DOI 和 References_DOI 写入 CSV 文件
        writer.writerow([doi, references])

        # 输出进度
        print(f"Processed row {i + 1} out of {df.shape[0]}.")

print("All rows processed and saved.")
