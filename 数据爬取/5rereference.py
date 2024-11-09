import pandas as pd
import requests
import csv
import time

# 输入和输出文件路径
input_file = '新版processed_with_references补全.csv'
output_file = '新版processed_with_references补全更新.csv'

# 读取已处理的 CSV 文件
df = pd.read_csv(input_file, encoding='utf-8')


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


# 打开输出文件，并准备逐行写入
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # 写入表头
    writer.writerow(df.columns)

    # 逐行处理数据
    for i, row in df.iterrows():
        doi = row['DOI']

        # 如果 References_DOI 列不为 "ERROR"，直接写入
        if row['References_DOI'] != "ERROR":
            writer.writerow(row)
            continue

        # 重新获取引用信息
        references = get_references_doi(doi)

        # 更新 References_DOI 列的值
        row['References_DOI'] = references

        # 写入更新后的行数据
        writer.writerow(row)

        # 输出进度
        print(f"Reprocessed row {i + 1} out of {df.shape[0]}.")


print("All ERROR rows reprocessed and saved.")
