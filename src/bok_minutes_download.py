import os
import sys
import logging
import platform
import re
import datetime as dt

import requests
import pandas as pd

from bs4 import BeautifulSoup

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)


logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info(f"python version => {sys.version}")
logger.info(f"platform system => {platform.system()}")


def get_minutes_file(page_addr: str, mdate: dt.datetime, rdate: dt.datetime):
    """
    의사록 파일 다운로드 함수
    :param str page_addr: 의사록 페이지 URL 주소
    :param datetime.datetime mdate: 수정일자
    :param datetime.datetime rdate: 등록일자

    """
    file_header = f'{pjt_home_path}/data/minutes/hwp/KO_'
    prefix_addr = "http://bok.or.kr"

    page = requests.get(page_addr, timeout=300)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find('dl', class_='down').find_all('a')

    for link in links:
        filename = link.get_text()
        filename = filename.replace('\r', '').replace('\t', '').replace('\n', '')
        if filename[-3:] == 'hwp':
            filename = mdate.strftime('%Y%m%d') + "_" + rdate.strftime('%Y%m%d') + '.hwp'
            file_addr = prefix_addr + link["href"]
            file_res = requests.get(file_addr)

            filepath = file_header + filename
            with open(filepath, 'wb') as f:
                f.write(file_res.content)

            logger.info(f"{filename} {file_addr}")


def get_minutes_list(from_date: str ='20171010'):
    """
    등록일(radte) 과  from_date 보다 미래인 날만 의사록 데이터 링크, mdate, rdate records list 반환
    rss html의 description 부분에 공란으로 되어 있어 기간별 구분 (rdate > 17 Sep 2019).
    # Because this is for the recent minutes, changed the address of the list to that of the rss feed of the BOK minutes.

    :param str from_date: 의사록 수집 시작일자 (YYYYMMDD)
    :return list: [guid, mdate, rdate]
    """

    from_date = dt.datetime.strptime(from_date, '%Y%m%d')
    url = 'https://www.bok.or.kr/portal/bbs/B0000245/news.rss?menuNo=200761'
    # url = 'https://www.bok.or.kr/portal/bbs/B0000245/list.do?menuNo=200761&pageIndex=1'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent': user_agent}
    page = requests.get(url, headers=headers)

    soup = BeautifulSoup(page.content, 'html.parser')
    brd_list = soup.find_all('item')

    minutes_list = list()

    for post in brd_list:
        pubdate = post.find('pubdate').get_text().strip()
        description = post.find('description').get_text().strip()
        guid = post.find('guid').get_text().strip()
        title = post.find('title').get_text().strip()

        rdate = pubdate[5:16]
        rdate = dt.datetime.strptime(rdate, '%d %b %Y')

        # 통화정책방향 의안 의사록만 분석
        if description.replace(' ', '').find('통화정책방향') >= 0 and rdate <= dt.datetime.strptime('2019-09-17', '%Y-%m-%d'):
            mdate = title[title.find(')(') + 2:-1]
            if mdate[-1] == '.':
                mdate = mdate[:-1]
            mdate = dt.datetime.strptime(mdate, '%Y.%m.%d')
            if mdate < from_date:
                break

            minutes_list.append([guid, mdate, rdate])
        else:
            mdate = title[title.find(')(') + 2:-1]
            if mdate[-1] == '.':
                mdate = mdate[:-1]
            mdate = dt.datetime.strptime(mdate, '%Y.%m.%d')
            if mdate < from_date:
                break

            minutes_list.append([guid, mdate, rdate])

        return minutes_list

def get_minutes_list_old():
    target_csv_file = 'minutes_link.csv'
    df_link = pd.read_csv(f"{pjt_home_path}/data/minutes/{target_csv_file}", encoding='utf-8', sep='|')

    for row in df_link.iterrows():
        # step2. df_link 각 행별로 mdate, rdate, page_url 을 추출
        mdate = row[1]['mdate']
        rdate = row[1]['rdate']
        page_url = row[1]['page_url']

        mdate = dt.datetime.strptime(mdate, "%Y-%m-%d")
        rdate = dt.datetime.strptime(rdate, "%Y-%m-%d")

        # step3. get_minutes_file(page_url, mdate, rate) 함수 실행
        get_minutes_file(page_url, mdate, rdate)

def get_minutes_target_date(target_date: str = '20171107'):
    """
    등록일(radte) 과  target_date 가 같은날만 의사록 다운로드 함수.
    rss html의 description 부분에 공란으로 되어 있어 기간별 구분 (rdate > 17 Sep 2019).
    # Because this is for the recent minutes, changed the address of the list to that of the rss feed of the BOK minutes.
    :param str target_date:  의사록 등록일자 (YYYYMMDD)

    """

    target_date = dt.datetime.strptime(target_date, '%Y%m%d')
    url = 'https://www.bok.or.kr/portal/bbs/B0000245/news.rss?menuNo=200761'
    # url = 'https://www.bok.or.kr/portal/bbs/B0000245/list.do?menuNo=200761&pageIndex=1'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent': user_agent}
    page = requests.get(url, headers=headers)

    soup = BeautifulSoup(page.content, 'html.parser')
    brdList = soup.find_all('item')

    for post in brdList:
        pubdate = post.find('pubdate').get_text().strip()
        description = post.find('description').get_text().strip()
        guid = post.find('guid').get_text().strip()
        title = post.find('title').get_text().strip()

        rdate = pubdate[5:16]
        rdate = dt.datetime.strptime(rdate, '%d %b %Y')

        if rdate == target_date:
            mdate = title[title.find(')(') + 2:-1]
            if mdate[-1] == '.':
                mdate = mdate[:-1]
            mdate = dt.datetime.strptime(mdate, '%Y.%m.%d')
            get_minutes_file(guid, mdate, rdate)

def get_minutes_data_from_list(minutes_list: list):
    """
    [guid, mdate, rdate] record 의 리스트를 입력 받아 해당 의사록 데이터 파일(hwp) 다운로드
    :param minutes_list:
    :return:
    """

    for item in minutes_list:
        guid = item[0]
        mdate = item[1]
        rdate = item[2]

        get_minutes_file(guid, mdate, rdate)


def convert_hwp_to_txt(from_date: str):
    """
    hwp 파일에서 txt 파일로 변환
    :param str from_date: 변환대상 hwp 파일의 등록일자(rdate)의 시작일자
    """
    list_minute_hwp_files = os.listdir(f'{pjt_home_path}/data/minutes/hwp')
    list_minute_hwp_files.sort()
    list_minute_hwp_files = [file_name for file_name in list_minute_hwp_files if file_name[-12:-4] >= from_date]

    for filename in list_minute_hwp_files:
        logger.info('convert to txt: %s' % filename)
        filename = filename[:-4]
        command_text = f"hwp5txt --output {pjt_home_path}/data/minutes/txt/{filename}.txt  {pjt_home_path}/data/minutes/hwp/{filename}.hwp"
        os.system(command_text)
        # 통화정책방향 선택
        target_txt_file = f'{pjt_home_path}/data/minutes/txt/{filename}.txt'
        if os.path.exists(target_txt_file):
            minutes = open(f'{pjt_home_path}/data/minutes/txt/{filename}.txt', encoding=u'utf-8').read()
            pos = re.search('〈의안 제[0-9]{1,3}호 ', minutes, re.MULTILINE)
            pos_end = pos.end() if pos else -1
            if minutes[pos_end + 1:pos_end + 34].find('통화정책방향') < 0:
                logger.info(f'remove txt: {filename}')
                # os.system(f'rm data/minutes/txt/{filename}.txt')
                os.remove(f'{pjt_home_path}/data/minutes/txt/{filename}.txt')
        else:
            logger.warning(f'fail to convert hwp to txt => {target_txt_file} !!')


def main(from_date: str):
    """
    메인함수
    :param str from_date: 의시록 수집 파일 시작일자 (YYYYMMDD)

    """
    logger.info(f"from_date => {from_date}")

    # tone 분석결과 데이터 로딩
    minutes_path = f'{pjt_home_path}/data/minutes/'
    df_tones_path = os.path.join(minutes_path, 'minutes_tones.csv')
    df_result = pd.read_csv(df_tones_path, encoding='utf-8', sep='|')
    last_result_mdate = df_result['mdate'].iloc[-1]
    last_result_mdate = dt.datetime.strptime(last_result_mdate[:10], "%Y-%m-%d")

    # 최신 금통위 의사록 목록
    minutes_list = get_minutes_list(from_date)
    last_mdate = minutes_list[0][1]
    logger.info(f"last_mdate-> {last_mdate.strftime('%Y-%m-%d')} last_result_mdate-> {last_result_mdate.strftime('%Y-%m-%d')}")
    # tone분석 일자와 최신 금통위 의사록 일자 비교 하여 의사록 데이터 다운로드
    if last_mdate > last_result_mdate:
        logger.info(f"last_mdate > last_result_mdate ==> download data & convert to txt...")
        get_minutes_data_from_list(minutes_list)
        convert_hwp_to_txt(from_date)
        logger.info("done!!")
    else:
        logger.info(f"last_mdate <= last_result_mdate ==> passing to download data & convert to txt!!")
    # get_minutes_list_old()
    # convert_hwp_to_txt(from_date)


if __name__ == "__main__":
    import argparse

    # 오늘 날짜에서 62일전 날짜를 기본값으로 설정
    now_dt = dt.datetime.now()
    default_date = (now_dt - dt.timedelta(days=62)).strftime("%Y%m%d")
    # default_date = '19990101'

    parser = argparse.ArgumentParser()
    parser.add_argument("from_date",
                        type=str,
                        default=default_date,
                        help=f"Date in YYYYMMDD format. Default is today ({default_date}).",
                        nargs='?'
                        )

    args = parser.parse_args()
    main(args.from_date)
