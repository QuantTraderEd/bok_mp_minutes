import os
import sys
import logging
import re
import time
import datetime as dt

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)

# Configure Selenium WebDriver for MacOS
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run browser in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Specify the path to chromedriver
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_mdate(text: str):
    # 정규식을 사용하여 날짜 추출
    pattern = r"\d{4}\.\d{1,2}\.\d{1,2}"
    match = re.search(pattern, text)

    formatted_date = ''

    # 결과 출력
    if match:
        extracted_date = match.group()
        # print("추출된 날짜:", extracted_date)

        # 날짜 형식 변환
        date_obj = dt.datetime.strptime(extracted_date, "%Y.%m.%d")
        formatted_date = date_obj.strftime("%Y-%m-%d")
        # print("변환된 날짜:", formatted_date)
    else:
        print("날짜를 찾을 수 없습니다.")

    return formatted_date

def extract_rate(row_text: str):

    if not '\n' in row_text: return ''

    row_text_list = row_text.split('\n')
    rdate_text = row_text_list[-2]
    date_obj = dt.datetime.strptime(rdate_text, "%Y.%m.%d")
    formatted_date = date_obj.strftime("%Y-%m-%d")

    return formatted_date

def extract_links(page_index: int = 1):
    url = f'https://www.bok.or.kr/portal/singl/newsData/list.do?pageIndex={page_index}&targetDepth=3&menuNo=201154&syncMenuChekKey=2&depthSubMain=&subMainAt=&searchCnd=1&searchKwd=&depth2=200038&depth3=201154&depth4=200789&date=&sdate=&edate=&sort=1&pageUnit=10'
    driver = setup_driver()
    driver.get(url)

    # Allow the page to load completely
    time.sleep(3)

    # Locate all the title elements containing href attributes
    elements = driver.find_elements(By.CLASS_NAME, 'title')
    elements = [element for element in elements if element.get_attribute('text') is not None]

    row_elements = driver.find_elements(By.CLASS_NAME, 'bbsRowCls')
    row_text_list = [element.text for element in row_elements if element.text != '']


    # Extract href values
    title_text_list = [element.get_attribute('text').replace('\n','').strip() for element in elements]
    href_link_list = [element.get_attribute('href') for element in elements]

    # Print the href values
    print("elements count=> ", len(elements))
    minutes_link_list = []
    for i in range(len(elements)-1):
        if title_text_list[i].startswith('금융통화위원회'):
            mdate = extract_mdate(title_text_list[i])
            rdate = extract_rate(row_text_list[i])
            minute_link = (title_text_list[i], mdate, rdate, href_link_list[i])
            minutes_link_list.append(minute_link)
            print(minute_link)

    # Close the browser
    driver.quit()

    return minutes_link_list

def main():
    minutes_link_list = []
    for page_index in range(11, 17):
        minutes_link_list += extract_links(page_index=page_index)

    columns_list = ['title', 'mdate', 'rdate', 'page_url']
    df = pd.DataFrame(minutes_link_list, columns=columns_list)
    df.to_csv(f'{pjt_home_path}/data/minutes/minutes_link_2019.csv', index=False, encoding='utf-8', sep='|')

if __name__ == "__main__":
    main()
