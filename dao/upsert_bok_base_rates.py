import os
import sys
import logging

import pandas as pd
import sqlite3

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)


logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)


def upsert_bok_base_rates(df_result: pd.DataFrame):
    """
    bok 기준금리 데이터 BOK_BASE_RATE 테이블에 upsert
    :param pd.DataFrame df_result:
    :return: upsert row count
    """
    # SQLite3 데이터베이스 연결
    conn = sqlite3.connect(pjt_home_path + '/db/bok_mp_minute_analytic.db')  # SQLite3 DB 연결
    cursor = conn.cursor()

    # 테이블 생성 쿼리 실행
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BOK_BASE_RATE (
        date TEXT PRIMARY KEY,
        base_rate TEXT
    )
    ''')

    # DataFrame 데이터를 dict 형태로 변환하여 일괄 SQL 처리
    records = df_result.to_dict(orient='records')

    cursor.executemany('''
    INSERT INTO bok_base_rate (date, base_rate) 
    VALUES (:date, :base_rate) 
    ON CONFLICT(date) DO UPDATE SET base_rate=excluded.base_rate;
    ''', records)

    # 변경된 행 수 출력
    upsert_rowcount = cursor.rowcount
    logger.info(f"업데이트되거나 삽입된 행 수: {cursor.rowcount}")

    # 연결 종료
    conn.commit()
    conn.close()

    return upsert_rowcount