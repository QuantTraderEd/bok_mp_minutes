import os
import sys
import site
import logging
import platform

import numpy as np
import pandas as pd
import sklearn
# import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path, os.pardir)
pjt_home_path = os.path.abspath(pjt_home_path)

site.addsitedir(pjt_home_path)

logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info(f"python version => {sys.version}")
logger.info(f"platform system => {platform.system()}")
logger.info(f"pandas version => {pd.__version__}")
logger.info(f"scikit-learn version => {sklearn.__version__}")


def main():

    # step 1. base_rate 데이터 로드
    df_bok_rate = pd.read_csv(pjt_home_path + '/data/base_rate_data.csv', encoding='utf-8', sep='|')
    df_bok_rate['date'] = pd.to_datetime(df_bok_rate['date'].astype(str))
    df_bok_rate = df_bok_rate.sort_values('date').reset_index(drop=True)

    # step 2. minute_daily_tone 데이터 로드
    df = pd.read_csv(pjt_home_path + '/data/minutes/minutes_tones.csv', encoding='utf-8', sep='|')

    df['mdate'] = pd.to_datetime(df['mdate'].str[:10])
    df['mdate'] = pd.to_datetime(df['mdate'].dt.date)

    df_bok_data = pd.merge(df_bok_rate, df, how='right', left_on='date', right_on='mdate')
    df_bok_data['base_rate'] = df_bok_data['base_rate'].ffill()
    df_bok_data['base_rate'] = df_bok_data['base_rate'].bfill()
    df_bok_data['date'] = df_bok_data['mdate']
    df_bok_data['D_BOK'] = df_bok_data['base_rate'].diff()
    df_bok_data = df_bok_data[['date', 'base_rate', 'D_BOK', 'tone_mkt', 'tone_lex']]

    # step 3. d_MP 값 생성
    df_bok_data['d_MP'] = df_bok_data['D_BOK'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # step 4. tone_mkt, tone_lex 값의 MACD 값 생성
    # To-Do: 파라미터 최적화 추가 테스트 가능..
    short_rng = 6
    long_rng = 24
    signal_rng = 6

    df_bok_data["tone_mkt_EMA_short"] = df_bok_data["tone_mkt"].ewm(span=short_rng, adjust=True).mean()
    df_bok_data["tone_mkt_EMA_long"] = df_bok_data["tone_mkt"].ewm(span=long_rng, adjust=True).mean()
    df_bok_data["tone_mkt_MACD"] = df_bok_data["tone_mkt_EMA_short"] - df_bok_data["tone_mkt_EMA_long"]
    df_bok_data["tone_mkt_MACD_sig"] = df_bok_data["tone_mkt_MACD"].ewm(span=signal_rng, adjust=True).mean()
    df_bok_data["tone_mkt_MACD_bool"] = df_bok_data["tone_mkt_MACD"] > df_bok_data["tone_mkt_MACD_sig"]

    df_bok_data["tone_lex_EMA_short"] = df_bok_data["tone_lex"].ewm(span=short_rng, adjust=True).mean()
    df_bok_data["tone_lex_EMA_long"] = df_bok_data["tone_lex"].ewm(span=long_rng, adjust=True).mean()
    df_bok_data["tone_lex_MACD"] = df_bok_data["tone_lex_EMA_short"] - df_bok_data["tone_lex_EMA_long"]
    df_bok_data["tone_lex_MACD_sig"] = df_bok_data["tone_lex_MACD"].ewm(span=signal_rng, adjust=True).mean()
    df_bok_data["tone_lex_MACD_bool"] = df_bok_data["tone_lex_MACD"] > df_bok_data["tone_lex_MACD_sig"]

    # step 5. d_MP_p1 (d_mp(t+1)) 생성
    df_bok_data.index = df_bok_data['date']
    df_bok_data['d_MP_p1'] = df_bok_data['d_MP'].shift(-1)

    # 금통위 의사록 발표후 다음 차후 금통위 까지 기간의 경우 아래 로직 패스
    # df_bok_data['d_MP_p1'].iloc[-1] = np.nan
    # df_bok_data['d_MP_p1'].iloc[-1] = 0
    logger.info(f"d_MP, d_MP_p1 chk => \n{df_bok_data[['d_MP', 'd_MP_p1']].iloc[-5:].to_markdown()}")

    df_bok_data1 = df_bok_data.dropna()
    df_bok_data1['d_MP_p1'] = df_bok_data1['d_MP_p1'].astype(int)

    # step 6. Random Forest Model Train
    # RF feature: tone_mkt, tone_lex, tone_mkt_MACD, tone_lex_MACD
    # 학습기간은 최근 15 건 이전 기간으로 학습하고 테스트는 최근 15건 (금리인상사이크) 구간으로 모델 테스트
    feature_list = ['d_MP', 'tone_mkt', 'tone_lex', 'tone_mkt_MACD', 'tone_lex_MACD']
    clf_model = RandomForestClassifier(n_estimators=5,
                                       min_samples_split=6,
                                       min_samples_leaf=4,
                                       max_features='sqrt',
                                       max_depth=24,
                                       bootstrap=True
                                       )
    logger.info(f"clf_model => \n{clf_model}")
    clf_model.fit(df_bok_data1[feature_list].iloc[:-15],
                  df_bok_data1['d_MP_p1'].iloc[:-15])
    model_score = clf_model.score(df_bok_data1[feature_list].iloc[:-15],
                                  df_bok_data1['d_MP_p1'].iloc[:-15])
    # logger.info(f"clf_model input feature cnt => {clf_model.n_features_in_}")
    logger.info(f"RF model score => {model_score}")

    # 예측결과와 실제결과 비교
    X = pd.DataFrame(df_bok_data1[feature_list].iloc[-14]).transpose()
    pred_result = clf_model.predict(X)
    pred_prob = clf_model.predict_proba(X)
    compare_result = (pred_result[0] == df_bok_data1.iloc[-14]['d_MP_p1'])
    logger.info(f"compare_result => {compare_result}")

    # 마지막 데이터 기반 예측 결과
    X = pd.DataFrame(df_bok_data1[feature_list].iloc[-1]).transpose()
    pred_result = clf_model.predict(X)
    pred_prob = clf_model.predict_proba(X)
    logger.info(f"prediction for latest {pred_result=} {pred_prob=}")

    # RandomForestClassifer Model Graph
    # estimator = clf_model.estimators_[4]
    # fig = plt.figure(figsize=(25, 10))
    # plot_tree(estimator,
    #           feature_names=feature_list,
    #           class_names='d_MP_p1',
    #           filled=True,
    #           impurity=True,
    #           rounded=True
    #           )

    pass


if __name__ == "__main__":
    main()