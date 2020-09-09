import logbook
import time
import re
import sys
import configparser
import os
import socket
import pickle

from db.NewsparserDatabaseHandler import NewsparserDatabaseHandler
from Lib.NewsHelper import Helper

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOption
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains


class NewsParserData(object):
    logger = None
    config = None
    driver = None

    def __init__(self, db, path_to_webdriver, config=None, logger=None):
        self.logger = logger
        self.logger.info("webdriver path: {}".format(path_to_webdriver))

        self.config = config

        chrome_options = ChromeOption()

        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)

        # ignore error proxy
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')

        # automatically dismiss prompt
        chrome_options.set_capability('unhandledPromptBehavior', 'dismiss')

        self.driver = webdriver.Chrome(path_to_webdriver, chrome_options=chrome_options)

        self.db = db

    def __del__(self):
        self.driver.quit()

    def openLink(self, URL):
        self.logger.info("opening URL: {}".format(URL))
        self.driver.get(URL)
        self.driver.implicitly_wait(10)
        time.sleep(5)
        self.logger.info("opening URL: DONE")

    def openWeb(self, input):

        news = self.db.get_source(input)

        #get ID, URL, SCROLL
        news_id = news[0]['id']
        url = news[0]['link']
        scroll = news[0]['scroll']

        self.openLink(url)
        self.logger.info("START Scrolling")
        Helper.scroll_down(self.driver, scroll)
        self.logger.info("FINISH Scrolling")

        return news_id, news

    def getLink(self, input):

        news_id, news = self.openWeb(input)

        #get CONTAINER, LINK
        get_container = news[0]['get_container']
        get_link = news[0]['get_link']

        self.logger.info("START getting news link")
        if news_id == 5:
            cont = self.driver.find_element_by_xpath(get_container)
            container = cont.find_elements_by_xpath(get_link)
        else:
            container = self.driver.find_elements_by_xpath(get_container)

        linked = []

        for i in range(len(container)):
            con = container[i]

            if news_id == 5:
                link_ = con.get_attribute('href')
            else:
                alink = con.find_element_by_xpath(get_link)
                link_ = alink.get_attribute('href')

            linked.append(link_)

        for i in linked:
            link_ = i
            if news_id == 1:
                link_ = self.linkFilterKompas(i)
            elif news_id == 2:
                link_ = self.linkFilterDetik(i)
            elif news_id == 4:
                link_ = self.linkFilterOkezone(i)
            elif news_id == 5:
                link_ = self.linkFilterCNN(i)

            if link_ is not None:
                self.openLink(link_)
                self.getElement(link_, news_id, news)

    def getElement(self, link, news_id, news):

        #get TITLE, DATE, CONTENT
        get_title = news[0]['get_title']
        get_date = news[0]['get_date']
        get_content = news[0]['get_content']
        get_page_container = news[0]['get_page_container']
        get_page = news[0]['get_page']

        title = self.driver.find_element_by_xpath(get_title).text

        self.logger.info("NEWS Title: {}".format(title))

        date_ = self.driver.find_element_by_xpath(get_date)

        if news_id == 4:
            date = Helper.toDate(date_.text)
        else:
            date = date_.get_attribute("content")

        self.logger.info("NEWS Date: {}".format(date))

        try:
            self.logger.info("getting NEWS content: first page")
            content_ = self.driver.find_element_by_xpath(get_content)
            p = content_.find_elements_by_xpath('.//p')
            news_content = ' '.join(item.text for item in p if "Baca juga:" not in item.text)
            try:
                self.logger.info("getting NEWS content: check if there is next page or not")
                page = self.driver.find_element_by_xpath(get_page_container)
                if page is not None:
                    paging = page.find_elements_by_xpath(get_page)
                    for i in range(len(paging) - 1):
                        self.logger.info("getting NEWS content: next page exist, getting content")
                        if news_id == 2:
                            link_ = link + "/" + str(i + 2)
                        elif news_id == 1 or news_id == 4:
                            link_ = link + "?page=" + str(i+2)
                        self.openLink(link_)
                        content_ = self.driver.find_element_by_xpath(get_content)
                        p = content_.find_elements_by_xpath('.//p')
                        news_content_ = ' '.join(item.text for item in p)
                        news_content = news_content + " " + news_content_

                self.logger.info("getting NEWS content: done getting page's content")
            except:
                self.logger.info("getting NEWS content: no next page")
        except:
            news_content = None

        self.logger.info("NEWS content: {}".format(news_content))

        self.db.insert_news(news_id, link, title, date, news_content)

    def linkFilterKompas(self, link):
        kpremium = "utm_source"
        kompas = ".kompas."

        if kpremium not in link:
            if kompas in link:
                return link

    def linkFilterCNN(self, link):
        video = "/video"

        if video not in link:
            return link

    def linkFilterOkezone(self, link):
        okezone = ".okezone."
        re_page = r"[?]\w+[=]\d"

        if okezone in link:
            link_ = re.sub(re_page, "", link)
            return link_

    def linkFilterDetik(self, link):
        dtv = "detiktv"
        re_detik = r"[?].*"

        if dtv not in link:
            if "php" not in link:
                link_ = re.sub(re_detik, "", link)
                return link_

    #def getDate(self, input):

    #def parsingNews(self, link):


class NewsParsing(object):
    config = None
    logger = None
    filename = ""
    iphelper = None
    db = None
    hostname = ''
    hostip = ''

    def init(self):

        self.filename, file_extension = os.path.splitext(os.path.basename(__file__))
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../config', 'config.ini')
        log_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../logs', '%s.log' % self.filename)

        # load config
        self.config = configparser.ConfigParser(strict=False, allow_no_value=True)
        self.config.read(config_file)

        # init logger
        logbook.set_datetime_format("local")
        self.logger = logbook.Logger(name=self.filename)
        format_string = '%s %s' % ('[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] {record.level_name}',
                                   '{record.channel}:{record.lineno}: {record.message}')
        if self.config.has_option('handler_stream_handler', 'verbose'):
            loghandler = logbook.StreamHandler(sys.stdout, level=self.config.get('Logger', 'level'), bubble=True,
                                               format_string=format_string)
            self.logger.handlers.append(loghandler)
            loghandler = logbook.TimedRotatingFileHandler(log_file, level=self.config.get('Logger', 'level'),
                                                          date_format='%Y%m%d', backup_count=5, bubble=True,
                                                          format_string=format_string)
            self.logger.handlers.append(loghandler)
        else:
            loghandler = logbook.TimedRotatingFileHandler(log_file, level=self.config.get('Logger', 'level'),
                                                          date_format='%Y%m%d', backup_count=5, bubble=True,
                                                          format_string=format_string)
            self.logger.handlers.append(loghandler)
        self.db = NewsparserDatabaseHandler.instantiate_from_configparser(self.config, self.logger)

    def run(self, link):
        start_time = time.time()
        self.init()
        self.hostname = socket.gethostname()
        self.hostip = socket.gethostbyname(self.hostname)
        self.logger.info("Starting {} on {}".format(type(self).__name__, self.hostname))
        self.newsParserData = NewsParserData(db=self.db, path_to_webdriver=self.config.get('Selenium', 'chromedriver_path'),
                                             config=self.config, logger=self.logger)

        self.newsParserData.getLink(link)

        self.logger.info("Finish %s" % self.filename)
        print("--- %s seconds ---" % (time.time() - start_time))