import os
import sys
import site
import logging
import datetime as dt

import pandas as pd

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)

site.addsitedir(pjt_home_path)

from src.send_mail import send_mail

logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

def main(pwd: str):

    # 분석결과 데이터 로딩
    minutes_path = f'{pjt_home_path}/data/minutes/'
    df_tones_path = os.path.join(minutes_path, 'minutes_tones.csv')
    df_result = pd.read_csv(df_tones_path, encoding='utf-8', sep='|')

    # 분석결과 메일 전송
    logger.info("send mail bok minutes analytics result...")
    main_text = f"""
                {df_result.iloc[-12:].to_markdown()}
                """
    send_mail('ggtt7@naver.com', pwd=pwd + "CH",
              to_mail_list=['ggtt7@naver.com', 'hj.edward.kim@kbfg.com'],
              mail_title='BOK minutes tone analytics',
              mail_text=main_text
              )

    logger.info("Done!!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pwd",
                        type=str,
                        default="",
                        nargs='?')

    args = parser.parse_args()
    main(args.pwd)
