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
    URL = "https://www.kompas.com/"

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

    def openWeb(self):
        self.openLink(self.URL)
        if "kompas" in self.URL :
            newslink = "kompas"
        self.logger.info("START Scrolling")
        Helper.scroll_down(self.driver, int(self.config.get(newslink, 'scroll')))
        self.logger.info("FINISH Scrolling")

        return newslink

    def getLink(self):

        newslink = self.openWeb()
        container = self.driver.find_elements_by_xpath(self.config.get(newslink, 'container'))

        self.logger.info("START getting news link")

        link = []

        for i in range(len(container)):
            con = container[i]

            alink = con.find_element_by_xpath(self.config.get(newslink, 'link'))
            link_ = alink.get_attribute('href')
            link.append(link_)

        for i in link:
            link_ = self.linkFilterKompas(i)
            if link_ is not None:
                self.openLink(link_)
                self.getElement(link_, newslink)

    def getElement(self, link, newslink):

        title = self.driver.find_element_by_xpath(self.config.get(newslink, 'title')).text

        self.logger.info("NEWS Title: {}".format(title))

        date_ = self.driver.find_element_by_xpath(self.config.get(newslink, 'date'))
        date = date_.get_attribute("content")

        self.logger.info("NEWS Date: {}".format(date))

        content_ = self.driver.find_element_by_xpath(self.config.get(newslink, 'content'))
        p = content_.find_elements_by_xpath('.//p')
        news_content = ' '.join(item.text for item in p)
        self.logger.info("getting NEWS content: first page")

        try:
            self.logger.info("getting NEWS content: check if there is next page or not")
            page = self.driver.find_element_by_xpath(self.config.get(newslink, 'page_container'))
            if page is not None:
                paging = page.find_elements_by_xpath(self.config.get(newslink, 'page'))
                for i in range(len(paging) - 1):
                    self.logger.info("getting NEWS content: next page exist, getting content")
                    link_ = link + "?page=" + str(i+2)
                    self.openLink(link_)
                    content_ = self.driver.find_element_by_xpath(self.config.get(newslink, 'content'))
                    p = content_.find_elements_by_xpath('.//p')
                    news_content_ = ' '.join(item.text for item in p)
                    news_content = news_content + " " + news_content_
        except:
            self.logger.info("getting NEWS content: no more next page.")

        self.logger.info("NEWS content: {}".format(news_content))

    def linkFilterKompas(self, link):
        kpremium = "utm_source"
        kompas = ".kompas."

        if kpremium not in link:
            if kompas in link:
                return link

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

    def run(self):
        start_time = time.time()
        self.init()
        self.hostname = socket.gethostname()
        self.hostip = socket.gethostbyname(self.hostname)
        self.logger.info("Starting {} on {}".format(type(self).__name__, self.hostname))
        self.newsParserData = NewsParserData(db=self.db, path_to_webdriver=self.config.get('Selenium', 'chromedriver_path'),
                                             config=self.config, logger=self.logger)

        self.newsParserData.getLink()

        self.logger.info("Finish %s" % self.filename)
        print("--- %s seconds ---" % (time.time() - start_time))