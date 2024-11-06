import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta


def onedayArticle(date=datetime(2018, 1, 1), file="default", start=0):
    # 设置要爬取的 URL
    search_date_str = date.strftime('%Y-%m-%d')
    next_day_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    #url = "https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-physics=y&classification-physics_archives=physics&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2020-01-01&date-to_date=2020-01-02&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"

    #下一行仅搜索狭义物理类
    #url = f"https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-physics=y&classification-physics_archives=physics&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={search_date_str}&date-to_date={next_day_str}&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"

    #下一行搜索整个物理类
    #url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-physics=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={search_date_str}&date-to_date={next_day_str}&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"

    #下一行搜索整个物理类，但是仅保留主分类是整个物理类
    url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-physics=y&classification-physics_archives=all&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range&date-from_date={search_date_str}&date-to_date={next_day_str}&date-date_type=submitted_date&abstracts=show&size=200&order=-announced_date_first"

    if start > 0:
        url = url + "&start=" + str(start)

    # 发送请求并获取页面内容
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    #print(soup)

    # 找到所有论文条目
    entries = soup.find_all('li', class_='arxiv-result')

    # 存储结果的列表
    results = []

    # 遍历每个 arxiv-result 元素
    for result in soup.select('li.arxiv-result'):
        # 提取信息
        arxiv_code = result.select_one('p.list-title a').text.strip()
        categories = [tag.text.strip() for tag in result.select('div.tags .tag')]
        title = result.select_one('p.title').text.strip()
        abstract = result.select_one('span.abstract-full').text.strip()

        # 提取提交日期
        submission_info = result.select_one('p.is-size-7').text.strip()
        submission_date = submission_info.split('Submitted')[1].split(';')[0].strip()  # 提取提交日期部分
        # 将提交日期转换为 YYYYMMDD 格式
        submission_date_obj = datetime.strptime(submission_date, '%d %B, %Y')
        formatted_submission_date = submission_date_obj.strftime('%Y%m%d')

        # 将提取的信息添加到列表中
        results.append({
            'arXiv Code': arxiv_code,
            'Categories': ', '.join(categories),
            'Title': title,
            'Abstract': abstract,
            'Submission Date': formatted_submission_date
        })


    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 保存为 CSV 文件
    if file == "default":
        file = search_date_str + ".csv"
    df.to_csv(file, mode='a', index=False, header=False, encoding='utf-8')
    #print("数据已成功保存到" + file)
    #if len(results) == 200:
     #   onedayArticle(date, file, start+200)
    return len(results)

# 主程序：循环遍历每一天并汇总结果
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 12, 31)

current_date = start_date
days_count = 0

# 创建 CSV 文件并写入表头
output_file = '广义物理类，不含交叉.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('arXiv Code,Categories,Title,Abstract,Submission Date\n')

# 循环遍历每一天
while current_date <= end_date:
    print(f"Fetching articles for date: {current_date.strftime('%Y-%m-%d')}",end='')
    new_num_articles = 0
    new_num_articles2 = 0
    num_articles = onedayArticle(current_date,output_file,0)
    if num_articles == 200:
       new_num_articles = onedayArticle(current_date, output_file, 200)
    if new_num_articles == 200:
        new_num_articles2 = onedayArticle(current_date, output_file, 400)

    total_articles = num_articles + new_num_articles + new_num_articles2
    if num_articles == 0 or total_articles == 600:
        print("The last date is" + current_date.strftime('%Y-%m-%d'))
        break
    print("This day"+str(total_articles))
    # 更新已汇总的天数
    days_count += 1
    current_date += timedelta(days=1)

