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


def upsert_bok_minute_sentence_tone_analytic_result(df_score: pd.DataFrame):
    """
    bok 기준금리 문장별 tone 데이터 BOK_MINUTE_SENTENCE_TONE_ANALYTIC 테이블에 upsert
    :param pd.DataFrame df_score:
    :return: upsert row count
    """

    # Database connection
    conn = sqlite3.connect(pjt_home_path + '/db/bok_mp_minute_analytic.db')
    cursor = conn.cursor()

    # Create table if it does not exist
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS BOK_MINUTE_SENTENCE_TONE_ANALYTIC (
        filename TEXT,
        mdate TEXT,
        rdate TEXT,
        section TEXT,
        sid INTEGER,
        sentence TEXT,
        tone_mkt REAL,
        tone_lex REAL,
        PRIMARY KEY (filename, sid)
    );
    """
    cursor.execute(table_creation_query)

    # Convert DataFrame to list of records
    records = df_score.to_dict(orient='records')

    # Upsert data into the table
    upsert_query = """
    INSERT INTO BOK_MINUTE_SENTENCE_TONE_ANALYTIC (filename, mdate, rdate, section, sid, sentence, tone_mkt, tone_lex)
    VALUES (:filename, :mdate, :rdate, :section, :sid, :sentence, :tone_mkt, :tone_lex)
    ON CONFLICT(filename, sid) DO UPDATE SET
        mdate = excluded.mdate,
        rdate = excluded.rdate,
        section = excluded.section,
        sentence = excluded.sentence,
        tone_mkt = excluded.tone_mkt,
        tone_lex = excluded.tone_lex;
    """

    rows_affected = cursor.executemany(upsert_query, records)
    upsert_rowcount = cursor.rowcount

    conn.commit()
    conn.close()

    # Output number of rows upserted
    logger.info(f"Number of rows upserted: {upsert_rowcount}")
    # logger.info(f"Number of rows affected: {rows_affected.rowcount}")

    return upsert_rowcount