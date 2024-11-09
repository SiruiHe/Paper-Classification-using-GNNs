import requests
import pandas as pd
import re


def preprocess_title(title):
    """去除标题中的格式化标记，标准化空格，并只保留字母"""
    title = re.sub(r'\$.*?\$', '', title)  # 去除所有 $...$ 内的 LaTeX 表达式
    title = re.sub(r'<.*?>', '', title)  # 去除所有 HTML 标签
    title = re.sub(r'[^a-zA-Z]', '', title)  # 只保留字母，删除所有非字母字符
    return title.strip().lower()


def is_subsequence(sub, main):
    """检查 sub 是否是 main 的子序列"""
    it = iter(main)
    return all(c in it for c in sub)


def compare_titles(title1, title2):
    """判断 title1 是否能通过删除字符得到 title2"""
    preprocessed_title1 = preprocess_title(title1)
    preprocessed_title2 = preprocess_title(title2)

    # 判断 preprocessed_title1 是否是 preprocessed_title2 的子序列，或者反之
    if is_subsequence(preprocessed_title1, preprocessed_title2):
        return True
    if is_subsequence(preprocessed_title2, preprocessed_title1):
        return True
    return False


def extract_authors(data):
    """提取作者信息，格式化为 '姓 名 ORCID' 的形式"""
    authors = data.get('message', {}).get('author', [])
    author_info = []
    for author in authors:
        family_name = author.get('family', '')
        given_name = author.get('given', '')
        orcid = author.get('ORCID', '')
        if orcid:
            author_info.append(f"{family_name} {given_name} {orcid}")
        else:
            author_info.append(f"{family_name} {given_name}")

    return '||'.join(author_info)  # 多个作者信息用 '||' 连接


input_file = 'processed_with_doi.csv'
output_file = 'comparison_results.csv'

# 使用 pandas 读取 CSV 文件
df = pd.read_csv(input_file)

# 新增字段列
df['Crossref Title'] = ''
df['Comparison'] = ''
df['Preprocessed Original Title'] = ''
df['Preprocessed Crossref Title'] = ''
df['Authors'] = ''  # 新增列用于保存作者信息

# 打开文件以追加模式写入
with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = pd.DataFrame(columns=df.columns)  # 用于写入表头
    writer.to_csv(outfile, index=False)

    # 处理每一行数据
    total_lines = len(df)
    for idx, row in df.iterrows():
        doi = row['DOI']
        original_title = row['Title']

        try:
            response = requests.get(f'https://api.crossref.org/works/{doi}')
            if response.status_code == 200:
                data = response.json()
                crossref_title = data['message']['title'][0]
                preprocessed_original_title = preprocess_title(original_title)
                preprocessed_crossref_title = preprocess_title(crossref_title)

                # 使用严格的子序列判断方法
                if compare_titles(preprocessed_original_title, preprocessed_crossref_title):
                    match = "Match"
                else:
                    match = "Mismatch"

                # 提取作者信息
                authors_info = extract_authors(data)
            else:
                crossref_title = 'Error fetching title'
                preprocessed_original_title = preprocess_title(original_title)
                preprocessed_crossref_title = 'N/A'
                match = 'Error'
                authors_info = 'Error fetching authors'
        except Exception as e:
            crossref_title = 'Exception occurred'
            preprocessed_original_title = preprocess_title(original_title)
            preprocessed_crossref_title = 'N/A'
            match = str(e)
            authors_info = 'Exception occurred'

        # 修改 Categories 字段
        categories = row['Categories'].split(',')
        modified_categories = ','.join(categories[:-2]) if len(categories) > 2 else row['Categories']

        # 更新 DataFrame
        row['Crossref Title'] = crossref_title
        row['Comparison'] = match
        row['Preprocessed Original Title'] = preprocessed_original_title
        row['Preprocessed Crossref Title'] = preprocessed_crossref_title
        row['Authors'] = authors_info  # 保存作者信息
        row['Categories'] = modified_categories

        # 写入当前行
        row.to_frame().T.to_csv(outfile, index=False, header=False, mode='a', encoding='utf-8')

        # 打印进度
        print(f"Progress: {idx + 1}/{total_lines}")

print("Comparison completed and results saved to", output_file)
