import os
import sys
import site
import logging
import platform
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import sklearn
# import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score
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

bok_rate_cycle_v1 = """
상승기_시작  상승기_종료  하락기_시작  하락기_종료
2000-01-06  2001-01-11  2001-02-08  2005-09-08
2005-10-11  2008-09-11  2008-10-09  2010-06-10
"""

def random_forest_model_simple_train(df_bok_data1: pd.DataFrame, feature_list: list):
    # 학습기간은 최근 15 건 이전 기간으로 학습하고 테스트는 최근 15건 (금리인상사이클) 구간으로 모델 테스트

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


def random_forest_model_tuning1(df_bok_data1: pd.DataFrame, feature_list: list):
    # Random Forest Model Tuning1
    logger.info("Random Forest Model Tuning1....")
    rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    calibrated_forest = CalibratedClassifierCV(rf, cv=5)
    param_grid = {'estimator__max_depth': [4, 8, 16],
                  'estimator__n_estimators': [10, 20, 50, 100, 200],
                  }
    search = GridSearchCV(calibrated_forest, param_grid, cv=3, n_jobs=-1)
    search.fit(df_bok_data1[feature_list], df_bok_data1['d_MP_p1'])

    logger.debug(sorted(search.cv_results_.keys()))
    logger.info(search.best_score_)
    logger.info(search.best_params_)
    logger.info(search.best_index_)
    logger.debug(search.cv_results_)
    logger.info("Random Forest Model Tuning1 Done!!!")

    best_model = search.best_estimator_

    return best_model


def random_forest_model_tuning2(df_bok_data1: pd.DataFrame, feature_list: list):
    logger.info("Random Forest Model Tuning2....")
    # Splitting the data into train-test split
    X = df_bok_data1[feature_list]
    Y = df_bok_data1['d_MP_p1']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=45)

    # Hyperparamter tuning using RandomizedsearchCV
    n_estimators = [5, 10, 20, 50]  # number of trees in the random forest
    max_features = ['auto', 'sqrt']  # number of feature in consideration at every split
    max_depth = [int(x) for x in np.linspace(4, 24, num=2)]  # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10]  # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False]  # method used to sample data points

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }

    rf_base = RandomForestClassifier(n_estimators=5, n_jobs=-1)
    rf_random_search = RandomizedSearchCV(estimator=rf_base,
                                          param_distributions=random_grid,
                                          n_iter=100,
                                          cv=5,
                                          verbose=2,
                                          random_state=35,
                                          n_jobs=-1)
    rf_random_search.fit(X_train, y_train)

    logger.debug(f'Tunging2 Random grid: {random_grid}')
    logger.info(f'Tunging2 Best Parameters: {rf_random_search.best_params_}')
    logger.info(f'Tunging2 Best Scores: {rf_random_search.best_score_}')
    logger.info("Random Forest Model Tuning2 Done!!!")

    # 예측 및 평가
    rf_cls_best_model = rf_random_search.best_estimator_
    probabilities = rf_cls_best_model.predict_proba(X_test)
    y_pred = rf_cls_best_model.predict(X_test)
    df_act_prd = pd.DataFrame([y_test.values, y_pred]).transpose()
    df_act_prd.index = y_test.index
    df_act_prd.columns = ['actual', 'predict']
    logger.info(f"test data set actual vs predict =>\n{df_act_prd}")

    # 성능 평가
    score = rf_cls_best_model.score(X_test, y_test)
    logloss = log_loss(y_test, probabilities)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Score: {score:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # 최근 데이터 적용 결과1 (최근 16개 적용 테스트)
    X_test_latest_16th = df_bok_data1[feature_list].iloc[-16:]
    y_test_latest_16th = df_bok_data1['d_MP_p1'].iloc[-16:]
    score = rf_cls_best_model.score(X_test_latest_16th, y_test_latest_16th)

    y_pred = rf_cls_best_model.predict(X_test_latest_16th)
    df_act_prd = pd.DataFrame([y_test_latest_16th.values, y_pred]).transpose()
    df_act_prd.index = y_test_latest_16th.index
    df_act_prd.columns = ['actual', 'predict']
    logger.info(f"test latest 16th data set actual vs predict =>\n{df_act_prd}")
    logger.info(f"Score for latest 16th: {score:.4f}")


    # 모델 저장
    with open(f'{pjt_home_path}/models/best_model.pkl', 'wb') as mdl_file:
        pickle.dump(rf_cls_best_model, mdl_file)

    logger.info("Best model saved as 'best_model.pkl'")


def random_forest_model_predict(df_bok_data1: pd.DataFrame, feature_list: list):
    if not os.path.exists(f'{pjt_home_path}/models/best_model.pkl'):
        logger.error('there is no model pkl file...!!!!')
        raise
    # 모델 로드
    logger.info(f"random forest model loading....")
    with open(f'{pjt_home_path}/models/best_model.pkl', 'rb') as mdl_file:
        rf_cls_best_model = pickle.load(mdl_file)

    # 최근 데이터 적용 결과 (모델 추론)
    logger.info(f"random forest model prediction....")
    X = pd.DataFrame(df_bok_data1[feature_list].iloc[-1]).transpose()
    pred_result = rf_cls_best_model.predict(X)
    pred_prob = rf_cls_best_model.predict_proba(X)
    logger.info(f"prediction for latest {pred_result=} {pred_prob=}")

    return pred_result, pred_prob


def main(base_date: str):
    """
    조건1: 데이터 셋에서 분석기준일 대응 신규 데이터가 있는 경유 모델 최적화 실행
    조건2: 일반 기준일에서는 모델 추론만 실행
    :param str base_date: 배치 실행 기준일자

    """

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
    # RF feature: d_MP, tone_mkt, tone_lex, tone_mkt_MACD, tone_lex_MACD
    feature_list = ['d_MP', 'tone_mkt', 'tone_lex', 'tone_mkt_MACD', 'tone_lex_MACD']

    # Random Forest Model Tuning2 (모델 학습 최적화)
    # To-Do: df_bok_data1 마지막 데이터 일자 와 base_date 일자 체크 하여 모델 최적화 실행
    random_forest_model_tuning2(df_bok_data1, feature_list)
    # 모델 추론
    pred_result, pred_prob = random_forest_model_predict(df_bok_data1, feature_list)
    pass


if __name__ == "__main__":
    import argparse

    # 오늘 날짜에서 1일전 날짜를 기본값으로 설정
    now_dt = dt.datetime.now()
    default_date = (now_dt - dt.timedelta(days=1)).strftime("%Y%m%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("base_date",
                        type=str,
                        default=default_date,
                        help=f"Date in YYYYMMDD format. Default is today ({default_date}).",
                        nargs='?'
                        )

    args = parser.parse_args()
    main(args.base_date)
