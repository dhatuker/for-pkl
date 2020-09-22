import logbook
import time
import sys
import configparser
import os
import socket

import gensim
from gensim.models import CoherenceModel
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from db.NewsparserDatabaseHandler import NewsparserDatabaseHandler

class PreprocessingData(object):
    logger = None
    config = None
    ts = None

    def __init__(self, db, config=None, logger=None):
        self.logger = logger
        self.config = config
        ts = time.localtime()
        self.time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", ts)
        self.db = db

        g = open("stopwords.txt", "r")
        self.gtext = g.read()
        g.close()

    def prepros(self):
        # today = str(date.today())
        today = '2020-09-20'
        data = list(self.db.get_article(today))
        stop_words = self.gtext.split("\n")
        stop_words.extend(['kompas','republika' ,'com' ,'co'])

        doc = list()

        for i in range(len(data)):
            if data[i]['content'] is not None:
                doc.append(data[i]['title']+" "+data[i]['content'])

        news_df = pd.DataFrame({'document': doc})

        # removing everything except alphabets`
        news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z0-9]", " ")

        # make all text lowercase
        news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

        # tokenization
        tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

        # remove stop-words
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(tokenized_doc, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[tokenized_doc], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

        data_words_bigrams = self.make_bigrams(tokenized_doc)

        data_join = self.join(data_words_bigrams)

        count_news = 0

        for news in data_join:
            count_news = count_news + 1
            self.logger.info("START input database")
            self.db.insert_prepro(self.time_stamp, count_news, news)

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self, texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def join(self, texts):
        texts_out = []
        for sent in texts:
            doc = " ".join(sent)
            texts_out.append(doc)
        return texts_out

class Proses(object):
    config = None
    logger = None
    db = None

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
        self.PreprocessingData = PreprocessingData(db=self.db, config=self.config, logger=self.logger)

        self.PreprocessingData.prepros()

        self.logger.info("Finish %s" % self.filename)
        print("--- %s seconds ---" % (time.time() - start_time))