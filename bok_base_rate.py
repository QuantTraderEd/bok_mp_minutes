import os
import sys
import site
import logging
import platform

import requests
import pandas as pd

from bs4 import BeautifulSoup

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path)
pjt_home_path = os.path.abspath(pjt_home_path)

site.addsitedir(pjt_home_path)

from dao.upsert_bok_base_rates import upsert_bok_base_rates


logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info(f"python version => {sys.version}")
logger.info(f"platform system => {platform.system()}")


def main():
    # URL 설정
    url = "https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643"

    # HTTP 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공했는지 확인

    # HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 테이블 데이터 찾기
    table = soup.find('table')  # 테이블 확인: 클래스 이름 제거 및 첫 번째 테이블 탐색
    if not table:
        raise ValueError("기준금리 데이터를 포함하는 테이블을 찾을 수 없습니다. HTML 구조를 확인하세요.")

    # 테이블 헤더와 데이터 추출
    headers = [header.text.strip() for header in table.find_all('th')]
    if not headers:
        raise ValueError("테이블 헤더를 찾을 수 없습니다. HTML 구조를 확인하세요.")

    data = []
    for row in table.find_all('tr')[1:]:  # 첫 번째 행은 헤더이므로 제외
        cols = [col.text.strip() for col in row.find_all('td')]
        if cols:
            data.append(cols)

    if not data:
        raise ValueError("테이블 데이터가 비어 있습니다. HTML 구조를 확인하세요.")

    # DataFrame 생성
    df = pd.DataFrame(data, columns=['year', 'month_day', 'base_rate'])

    # 년, 월, 일을 조합하여 새로운 YYYYMMDD 포맷 날짜 컬럼 생성
    df['date'] = (df['year'] + df['month_day'].str.zfill(2)).str.replace('[^\d]', '', regex=True)

    # 새로운 DataFrame: 날짜 컬럼과 데이터의 마지막 컬럼만 남김
    df_result = df[['date', 'base_rate']]
    logger.info('df_result=>\n' + df_result.iloc[:5].to_markdown())

    # sqlite3 db 에 upsert
    upsert_bok_base_rates(df_result)

    # CSV 파일로 저장
    df_result.to_csv(pjt_home_path + '/data/base_rate_data.csv', index=False, encoding='utf-8', sep='|')
    logger.info("기준금리 데이터를 'base_rate_data.csv' 파일로 저장했습니다.")


if __name__ == "__main__":
    import argparse

    main()
