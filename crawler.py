import pandas as pd
import requests
import csv
import time
import sys
import os
from time import sleep
from random import uniform
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock

# 添加命令行参数处理
start_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# 读取文件
input_file = 'processed_with_doi.csv'
output_file = 'processed_with_references.csv'

df = pd.read_csv(input_file)

# 检查是否存在输出文件并读取已处理的数据
processed_dois = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        processed_dois = {row[0] for row in reader}
    
    # 如果存在输出文件，则以追加模式打开
    write_mode = 'a'
    write_header = False
else:
    write_mode = 'w'
    write_header = True

# 添加一个结果队列和文件写入锁
result_queue = Queue()
file_lock = Lock()

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
def get_references_doi(doi, max_retries=3):
    if pd.isna(doi):
        return "No References"

    retries = 0
    while retries < max_retries:
        try:
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            references = data.get("message", {}).get("reference", [])
            reference_dois = [ref.get("DOI") for ref in references if "DOI" in ref]

            if not reference_dois:
                return "No References"

            return '||'.join(reference_dois)
            
        except requests.RequestException as e:
            retries += 1
            if retries == max_retries:
                print(f"Failed after {max_retries} attempts for DOI {doi}: {e}")
                return "ERROR"
            
            # 计算退避时间：1秒、2秒、4秒
            wait_time = (2 ** (retries - 1)) + uniform(0, 1)
            print(f"Attempt {retries} failed for DOI {doi}. Retrying in {wait_time:.1f} seconds...")
            sleep(wait_time)

def process_doi(row_data):
    """处理单个DOI的函数"""
    i, row = row_data
    doi = escape_csv_field(row['DOI'])
    
    # 跳过已处理的DOI
    if doi in processed_dois:
        print(f"Skipping already processed DOI at row {i + 1}")
        return None
        
    references = get_references_doi(doi)
    sleep(uniform(0.5, 1.5))  # 随机延时
    
    return (i, doi, references)

def write_results():
    """写入结果到文件"""
    with open(output_file, mode=write_mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['DOI', 'References_DOI'])

        while True:
            result = result_queue.get()
            if result is None:  # 结束信号
                break
                
            i, doi, references = result
            with file_lock:
                writer.writerow([doi, references])
                f.flush()
                print(f"Processed row {i + 1} out of {df.shape[0]}.")

# 主处理逻辑
def main():
    # 创建线程池
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(process_doi, (i, row)): i 
            for i, row in df.iloc[start_index:].iterrows()
        }
        
        # 按顺序获取结果并放入队列
        for future in future_to_row:
            result = future.result()
            if result:
                result_queue.put(result)
    
    # 发送结束信号
    result_queue.put(None)

if __name__ == '__main__':
    # 启动写入线程
    from threading import Thread
    writer_thread = Thread(target=write_results)
    writer_thread.start()
    
    # 运行主处理逻辑
    main()
    
    # 等待写入线程完成
    writer_thread.join()
    
    print("All rows processed and saved.")