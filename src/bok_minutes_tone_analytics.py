import os
import sys
import site
import logging
import platform
import datetime as dt

import pandas as pd
# import ekonlpy
from ekonlpy.sentiment import MPKO

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)

site.addsitedir(pjt_home_path)

from dao.upsert_bok_minute_sentence_tone_analytic import  upsert_bok_minute_sentence_tone_analytic_result
from dao.upsert_bok_minute_daily_tone_analytic import upsert_bok_minute_daily_tone_analytic_result

logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info(f"python version => {sys.version}")
logger.info(f"platform system => {platform.system()}")


def calc_polarity(scores):
    eps = 1e-6
    pos_score = [1 for s in scores if s > 0]
    neg_score = [-1 for s in scores if s < 0]

    s_pos = sum(pos_score)
    s_neg = sum(neg_score)

    s_pol = ((s_pos + s_neg) * 1.0 /
             ((s_pos - s_neg) + eps))

    return s_pol


def main():
    """
    BOK 금통위 의사록 sentimemt 분석 실행, 결과 저장 메인 함수

    """
    # step 1. minutes 데이터 로딩
    logger.info("loading minutes data...")
    minutes_path = f'{pjt_home_path}/data/minutes/'
    df_minutes_path = os.path.join(minutes_path, 'minutes.csv')
    df = pd.read_csv(df_minutes_path, encoding='utf-8', sep="|")

    df_texts_path = os.path.join(minutes_path, 'minutes.csv')
    df_tones_path = os.path.join(minutes_path, 'minutes_tones.csv')

    df_texts = pd.read_csv(df_texts_path, index_col=None, encoding='utf-8', sep="|")

    # step 2. 문장별 감성 분석
    logger.info("start sentence sentimental analysis...")

    # Korean Monetary Policy Dictionary (MPKO)
    mpko_mkt = MPKO(kind=0)
    mpko_lex = MPKO(kind=1)

    scores = []
    for row in df_texts.to_records(index=False):
        filename, mdate, rdate, section, sid, sentence = row
        tokens = mpko_lex.tokenize(sentence)
        tone_lex = mpko_lex.get_score(tokens)['Polarity']
        tokens = mpko_mkt.tokenize(sentence)
        tone_mkt = mpko_mkt.get_score(tokens)['Polarity']
        score = (filename, mdate, rdate, section, sid, sentence, tone_mkt, tone_lex)
        scores.append(score)

        # msg = f"{mdate} {rdate} {sentence} {tone_mkt} {tone_lex}"
        # logger.debug(msg)

    logger.info("sentence sentimental analysis done!!")

    # step 3. 분석결과 파일 저장
    logger.info("grouping analytics result by daily and save analytics result...")
    df_score = pd.DataFrame(scores, columns=['filename', 'mdate', 'rdate', 'section', 'sid', 'sentence', 'tone_mkt', 'tone_lex'])
    df_score.to_csv(minutes_path + 'minutes_score.csv',encoding='utf-8', index=False, sep='|')
    row_cnt = upsert_bok_minute_sentence_tone_analytic_result(df_score)
    logger.info("save sentence tone data!!")


    key_cols = ['mdate']
    tone_cols = ['tone_mkt', 'tone_lex']
    df_result = df_score.groupby(key_cols)[tone_cols].agg(lambda x: calc_polarity(x)).reset_index()
    logger.info(f"df_result=> \n{df_result.iloc[-12:].to_markdown()}")
    df_result.to_csv(df_tones_path, encoding='utf-8', index=False, sep='|')
    row_cnt = upsert_bok_minute_daily_tone_analytic_result(df_result)
    logger.info("save daily tone date!!")

    logger.info("Done!!")


if __name__ == "__main__":
    import argparse

    # 오늘 날짜에서 365일전 날짜를 기본값으로 설정
    now_dt = dt.datetime.now()
    default_date = (now_dt - dt.timedelta(days=365)).strftime("%Y%m%d")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("from_date",
    #                     type=str,
    #                     default=default_date,
    #                     help=f"Date in YYYYMMDD format. Default is today ({default_date}).",
    #                     nargs='?'
    #                     )
    # parser.add_argument("pwd",
    #                     type=str,
    #                     default="",
    #                     nargs='?')
    #
    # args = parser.parse_args()
    # main(args.pwd)
    main()
