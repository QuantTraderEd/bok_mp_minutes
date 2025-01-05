import os
import sys
import platform
import re
import datetime as dt
import pprint

import requests
import pandas as pd

from bs4 import BeautifulSoup

print(sys.version)
print(platform.system())


def get_minutes_file(page_addr, mdate, rdate):
    file_header = 'data/minutes/hwp/KO_'
    prefix_addr = "http://bok.or.kr"

    page = requests.get(page_addr)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find('div', class_='addfile').find_all('a')
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

            print(filename, file_addr)


# Because this is for the recent minutes, changed the address of the list to that of the rss feed of the BOK minutes.

def get_minutes_list(from_date='20180101', description_chk=True):
    from_date = dt.datetime.strptime(from_date, '%Y%m%d')
    url = 'https://www.bok.or.kr/portal/bbs/B0000245/news.rss?menuNo=200761'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent': user_agent}
    page = requests.get(url, headers=headers)

    soup = BeautifulSoup(page.content, 'html.parser')
    brd_list = soup.find_all('item')

    for post in brd_list:
        pubdate = post.find('pubdate').get_text().strip()
        description = post.find('description').get_text().strip()
        guid = post.find('guid').get_text().strip()
        title = post.find('title').get_text().strip()


        mdate = title[title.find(')(') + 2:-1]
        if mdate[-1] == '.':
            mdate = mdate[:-1]
        mdate = dt.datetime.strptime(mdate, '%Y.%m.%d')

        rdate = pubdate[5:16]
        rdate = dt.datetime.strptime(rdate, '%d %b %Y')

        print(description)
        print(mdate, rdate, guid)

        if description.replace(' ', '').find('통화정책방향') >= 0 and description_chk:

            if mdate < from_date:
                break
            get_minutes_file(guid, mdate, rdate)

        elif not description_chk:

            if mdate < from_date:
                break
            get_minutes_file(guid, mdate, rdate)


# Cleasing texts and make a corpus
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

    pos = re.search('(.?국내외\s?경제\s?동향.?과 관련하여,?|\(가\).+경제전망.*|\(가\).+경제상황 평가.*|\(가\) 국내외 경제동향 및 평가)\n?\s*일부 위원은', minutes, re.MULTILINE)
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

    print(' ==> text processing completed: {}'.format(filename))

    doc = []

    if any(section_texts):
        for s, (section, sentences) in enumerate(zip(sections, section_texts)):
            for p, text in enumerate(sentences):
                row = (filename, mdate, rdate, section, p, text)
                doc.append(row)
    else:
        print('Empty!')

    return doc, section_titles


def main(from_date: str):

    get_minutes_list(from_date=from_date, description_chk=False)

    filepath = './data/minutes/'
    hwp_filelist = os.listdir(filepath + 'hwp')
    for hwp_filename in hwp_filelist:
        target_filename = hwp_filename[:-4] + '.txt'
        convert_cmd = f"hwp5txt --output {filepath + 'txt/' + target_filename} {filepath + 'hwp/' + hwp_filename}"
        os.system(convert_cmd)
        print(convert_cmd)


def list_minute_files(path):
    path = os.path.join(path, 'txt')
    for i, file in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file)
        print('--processing {0}th minutes'.format(i+1))
        yield file_path




def test():
    file_name = './data/minutes/txt/KO_20230713_20230801.txt'
    minutes_path = './data/minutes/'
    files = list_minute_files(minutes_path)
    docs = []
    section_names = ['국내외 경제동향',
                     '외환․국제금융 동향',
                     '금융시장 동향',
                     '통화정책방향',
                     '위원별 의견 개진',
                     '정부측 열석자 발언',
                     ]
    for file in files:
        doc, section_titles = preprocess_minutes(file_path=file)
        # pprint.pprint(dict(zip(section_names, section_titles)))
        for i, title in enumerate(section_titles):
            print(section_names[i], ':', title.replace('\n', ' '))


if __name__ == "__main__":
    import argparse

    # 오늘 날짜를 기본값으로 설정
    default_date = dt.datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("from_date",
                        type=str,
                        default=default_date,
                        help=f"Date in YYYYMMDD format. Default is today ({default_date}).",
                        nargs='?'
                        )

    args = parser.parse_args()
    main(args.from_date)
