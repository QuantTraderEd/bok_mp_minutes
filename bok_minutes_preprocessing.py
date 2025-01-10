import os
import sys
import logging
import platform
import re
import datetime as dt

import requests
import pandas as pd

src_path = os.path.dirname(__file__)
pjt_home_path = os.path.join(src_path)
pjt_home_path = os.path.abspath(pjt_home_path)


logger = logging.getLogger(__file__)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)
stream_log = logging.StreamHandler(sys.stdout)
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info(f"python version => {sys.version}")
logger.info(f"platform system => {platform.system()}")


def tidy_sentences(section):
    sentence_enders = re.compile(r'((?<=[함음됨임봄짐움])(\s*\n|\.|;)|(?<=다)\.)\s*')
    splits = list((m.start(), m.end()) for m in re.finditer(sentence_enders, section))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits]
    sentences = [section[start:end] for start, end in zip(starts[:-1], ends)]
    for i, s in enumerate(sentences):
        sentences[i] = (s.replace('\n', ' ').replace(' ', ' ')) + '.'

    text = '\n'.join(sentences) if len(sentences) > 0 else ''
    return sentences, text


def preprocess_minutes(file_path):
    filename = file_path[-24:-4]
    mdate = dt.datetime.strptime(file_path[-21:-13], '%Y%m%d') + dt.timedelta(hours=10)
    rdate = dt.datetime.strptime(file_path[-12:-4], '%Y%m%d') + dt.timedelta(hours=16)

    print('open file: {}'.format(file_path))
    minutes = open(file_path, encoding=u'utf-8').read()

    # (가) 경제상황 평가 추가 (23.7.13)
    pos = re.search('(.?국내외\s?경제\s?동향.?과 관련하여,?|\(가\).+경제전망.*|\(가\).+경제상황.*|\(가\) 국내외 경제동향 및 평가)\n?\s*일부 위원은', minutes,
                    re.MULTILINE)
    s1 = pos.start() if pos else -1
    s1_end = pos.end() if pos else -1
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여,?|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*(일부 위원은|대부분의 위원들은)', minutes,
                    re.MULTILINE)
    s2 = pos.start() if pos else -1
    s2_end = pos.end() if pos else -1
    pos = re.search('(.?금융시장\s?동향.?과 관련하여,?|\(다\) 금융시장\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    s3 = pos.start() if pos else -1
    s3_end = pos.end() if pos else -1
    # pos = re.search('((\((다|라)\) )?.?통화정책방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에 관해 다음과 같은 토론이 있었음.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책.?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견\s?교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes,
                    re.MULTILINE)
    s4 = pos.start() if pos else -1
    s4_end = pos.end() if pos else -1
    pos = re.search('(\(4\) 정부측 열석자 발언.*)\n?', minutes, re.MULTILINE)
    s5 = pos.start() if pos else -1
    s5_end = pos.end() if pos else -1
    pos = re.search('(\(.*\) 한국은행 기준금리 결정에 관한 위원별 의견\s?개진|이상과 같은 토론에 이어 .* 관한 위원별 의견개진이 있었음.*)\n?', minutes,
                    re.MULTILINE)
    s6 = pos.start() if pos else -1
    s6_end = pos.end() if pos else -1
    # pos = re.search('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    # s7 = pos.start() if pos else -1
    positer = re.finditer('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    s7 = [pos.start() for pos in positer if pos.start() > s6]
    s7 = s7[0] if s7 else -1

    # 국내외 경제동향
    bos = s1
    eos = s2
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section1, section1_txt = tidy_sentences(section)
    section1_title = minutes[s1:s1_end] if s1 >= 0 and s1_end >= 0 else ''

    # 외환․국제금융 동향
    bos = s2
    eos = s3 if s3 >= 0 else s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section2, section2_txt = tidy_sentences(section)
    section2_title = minutes[s2:s2_end] if s2 >= 0 and s2_end >= 0 else ''

    # 금융시장 동향
    bos = s3
    eos = s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section3, section3_txt = tidy_sentences(section)
    section3_title = minutes[s3:s3_end] if s3 >= 0 and s3_end >= 0 else ''

    # 통화정책방향
    bos = s4
    eos = s5 if s5 >= 0 else s6 if s6 >= 0 else s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section4, section4_txt = tidy_sentences(section)
    section4_title = minutes[s4:s4_end] if s4 >= 0 and s4_end >= 0 else ''

    # 위원별 의견 개진
    bos = s6
    eos = s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section5, section5_txt = tidy_sentences(section)
    section5_title = minutes[s6:s6_end] if s6 >= 0 and s6_end >= 0 else ''

    # 정부측 열석자 발언
    bos = s5
    eos = s6
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('정부측 열석자 발언', section, re.MULTILINE)
    bos = pos.end() + 1 if pos else -1
    section = section[bos:] if bos >= 0 else section
    section6, section6_txt = tidy_sentences(section)
    section6_title = minutes[s5:s5_end] if s5 >= 0 and s5_end >= 0 else ''

    sections = ['Economic Situation', 'Foreign Currency', 'Financial Markets',
                'Monetary Policy', 'Participants’ Views', 'Government’s View']
    section_texts = (section1, section2, section3, section4, section5, section6)
    section_titles = [section1_title, section2_title, section3_title, section4_title, section5_title, section6_title]

    logger.info(' ==> text processing completed: {}'.format(filename))

    doc = []

    if any(section_texts):
        for s, (section, sentences) in enumerate(zip(sections, section_texts)):
            for p, text in enumerate(sentences):
                row = (filename, mdate, rdate, section, p, text)
                doc.append(row)
    else:
        logger.info('Empty!')

    return doc, section_titles


def preprocess_minutes_20200331(file_path):
    ## 2020년 3월 16일 임시금통위 전용 테스트
    minutes_path = './data/minutes'
    file = minutes_path + '/txt/' + 'KO_20200316_20200331.txt'

    filename = file_path[-24:-4]
    mdate = datetime.strptime(file_path[-21:-13], '%Y%m%d') + timedelta(hours=10)
    rdate = datetime.strptime(file_path[-12:-4], '%Y%m%d') + timedelta(hours=16)

    print('open file: {}'.format(file_path))
    minutes = open(file_path, encoding=u'utf-8').read()

    pos = re.search('(.?국내외\s?경제\s?동향.?과 관련하여,?|\(가\).+경제전망.*|\(가\) 국내외 경제동향 및 평가)\n?\s*일부 위원은', minutes, re.MULTILINE)
    s1 = pos.start() if pos else -1
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여,?|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*(일부 위원은|대부분의 위원들은)', minutes,
                    re.MULTILINE)
    s2 = pos.start() if pos else -1
    pos = re.search('(.?금융시장\s?동향.?과 관련하여,?|\(다\) 금융시장\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    s3 = pos.start() if pos else -1
    # pos = re.search('((\((다|라)\) )?.?통화정책방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에 관해 다음과 같은 토론이 있었음.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책.?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견\s?교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes,
                    re.MULTILINE)
    s4 = pos.start() if pos else -1
    pos = re.search('(\(4\) 정부측 열석자 발언.*)\n?', minutes, re.MULTILINE)
    s5 = pos.start() if pos else -1

    # pos = re.search('(\(.*\) 한국은행 기준금리 결정에 관한 위원별 의견\s?개진|이상과 같은 토론에 이어 .* 관한 위원별 의견개진이 있었음.*)\n?', minutes,
    #                 re.MULTILINE)
    # (2) 위원 토의내용 \n 위원별로 한국은행 기준금리 조정에 대한 의견을 개진하였음. (3.16)
    # (4) 한국은행 기준금리 결정에 관한 위원별 의견 개진\n  당일 개최된 본회의에서는 한국은행 기준금리 결정에 관한 위원별 의견 개진이 있었음. (4.9)
    pos = re.search('(\(.*\) 위원 토의내용|이상과 같은 토론에 이어 .* 관한 위원별 의견개진이 있었음.*)\n?', minutes, re.MULTILINE)

    s6 = pos.start() if pos else -1
    # pos = re.search('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    # s7 = pos.start() if pos else -1
    positer = re.finditer('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    s7 = [pos.start() for pos in positer if pos.start() > s6]
    s7 = s7[0] if s7 else -1

    # 국내외 경제동향
    bos = s1
    eos = s2
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section1, section1_txt = tidy_sentences(section)

    # 외환․국제금융 동향
    bos = s2
    eos = s3 if s3 >= 0 else s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section2, section2_txt = tidy_sentences(section)

    # 금융시장 동향
    bos = s3
    eos = s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section3, section3_txt = tidy_sentences(section)

    # 통화정책방향
    bos = s4
    eos = s5 if s5 >= 0 else s6 if s6 >= 0 else s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section4, section4_txt = tidy_sentences(section)

    # 위원별 의견 개진
    bos = s6
    eos = s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section5, section5_txt = tidy_sentences(section)

    # 정부측 열석자 발언
    bos = s5
    eos = s6
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('정부측 열석자 발언', section, re.MULTILINE)
    bos = pos.end() + 1 if pos else -1
    section = section[bos:] if bos >= 0 else section
    section6, section6_txt = tidy_sentences(section)

    sections = ['Economic Situation', 'Foreign Currency', 'Financial Markets',
                'Monetary Policy', 'Participants’ Views', 'Government’s View']
    section_texts = (section1, section2, section3, section4, section5, section6)

    print(' ==> text processing completed: {}'.format(filename))

    doc = []

    if any(section_texts):
        for s, (section, sentences) in enumerate(zip(sections, section_texts)):
            for p, text in enumerate(sentences):
                row = (filename, mdate, rdate, section, p, text)
                doc.append(row)
    else:
        print('Empty!')

    return doc


def list_minute_files(path):
    path = os.path.join(path, 'txt')
    file_list = os.listdir(path)
    file_list.sort()
    for i, file in enumerate(file_list):
        file_path = os.path.join(path, file)
        logger.info(f'--processing {i+1}th minutes: {file}')
        yield file_path


def main():

    # step 1: preprocessing 처리 할 파일 목록 리스트 생성
    minutes_path = './data/minutes'
    files = list_minute_files(minutes_path)
    docs = []

    section_names = ['국내외 경제동향',
                     '외환․국제금융 동향',
                     '금융시장 동향',
                     '통화정책방향',
                     '위원별 의견 개진',
                     '정부측 열석자 발언',
                     ]

    # step 2: txt 파일 별 preprocessing 실행
    for file in files:
        if file[-4:] != '.txt': continue
        if file == minutes_path + '/txt/' + 'KO_20200316_20200331.txt':  # 20.03.16 임시 금통위
            doc = preprocess_minutes_20200331(file)
        else:
            doc, section_titles = preprocess_minutes(file)
            for i, title in enumerate(section_titles):
                title = title.replace('\n', ' ')
                logger.info(f"{section_names[i]} : {title}")
        docs += doc

    # step 3: preprocessing 결과 파일 생성
    df = pd.DataFrame(docs, columns=['filename', 'mdate', 'rdate', 'section', 'sid', 'text'])
    df = df.dropna()
    df_minutes_path = os.path.join(minutes_path, 'minutes.csv')
    df_minutes_path_excel = os.path.join(minutes_path, 'minutes.xlsx')
    df.to_csv(df_minutes_path, encoding='utf-8', index=False, sep='|')
    df.to_excel(df_minutes_path_excel, index=False)


if __name__ == "__main__":
    import argparse

    # 오늘 날짜에서 30일전 날짜를 기본값으로 설정
    now_dt = dt.datetime.now()
    default_date = (now_dt - dt.timedelta(days=30)).strftime("%Y%m%d")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("from_date",
    #                     type=str,
    #                     default=default_date,
    #                     help=f"Date in YYYYMMDD format. Default is today ({default_date}).",
    #                     nargs='?'
    #                     )
    #
    # args = parser.parse_args()
    # main(args.from_date)
    main()
