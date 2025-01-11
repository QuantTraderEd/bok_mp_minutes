import os
import logging
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__file__)


def mail_server_login(mail_accnt='', pwd=''):
    """
    mail_server_login
    :param mail_accnt: ex: login@naver.com
    :param pwd:  ex: pwd
    :return: mail_server
    """
    mail_server = smtplib.SMTP('smtp.naver.com', 587)
    mail_server.ehlo()
    mail_server.starttls()
    mail_server.ehlo()
    # mail_server.starttls()

    mail_server.login(mail_accnt, pwd)

    return mail_server


def send_mail(mail_accnt: str, pwd: str, to_mail_list: list,
              mail_title: str, mail_text: str):

    mail_server = mail_server_login(mail_accnt, pwd)

    # 제목, 본문 작성
    msg = MIMEMultipart()
    msg['From'] = mail_accnt
    msg['To'] = ', '.join(to_mail_list)
    msg['Subject'] = mail_title
    msg.attach(MIMEText(mail_text, 'plain'))

    # 파일첨부 (파일 미첨부시 생략가능)
    # attachment = open('./data/%s' % target_filename, 'rb')
    # part = MIMEBase('application', 'octet-stream')
    # part.set_payload(attachment.read())
    # encoders.encode_base64(part)
    # filename = os.path.basename('./data/%s' % target_filename)
    # part.add_header('Content-Disposition', "attachment; filename= " + filename)
    # msg.attach(part)

    # 메일 전송
    mail_server.sendmail(mail_accnt, to_mail_list, msg.as_string())
    logger.info(f"send mail: {mail_title}")

    mail_server.quit()


if __name__ == "__main__":
    import getpass
    pwd = getpass.getpass('pwd')
    send_mail('login@naver.com', pwd, ['to@test.com'], 'test', 'test')