import logbook
import time
import numpy as np
import pandas as pd
import re
import sys
import configparser
import os
import socket

from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from db.NewsparserDatabaseHandler import NewsparserDatabaseHandler

class PreprocessingData(object):
    logger = None
    config = None
    driver = None

    def __init__(self, db, config=None, logger=None):
        self.logger = logger
        self.config = config
        self.db = db

        g = open("stopwords.txt", "r")
        self.gtext = g.read()
        g.close()

    def prepros(self):
        # today = str(date.today())
        today = '2020-09-08'
        data = list(self.db.get_article(today))
        stop_words = self.gtext.split("\n")

        doc = list()

        for i in range(len(data)):
            if data[i]['content'] is not None:
                doc.append(data[i]['title']+" "+data[i]['content'])

        news_df = pd.DataFrame({'document': doc})

        # removing everything except alphabets`
        news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")

        # removing null fields
        news_df = news_df[news_df['clean_doc'].notnull()]

        # make all text lowercase
        news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

        # tokenization
        tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

        # remove stop-words
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

        # de-tokenization
        detokenized_doc = []
        for i in range(len(tokenized_doc)):
            if i in tokenized_doc:
                t = ' '.join(tokenized_doc[i])
                detokenized_doc.append(t)

        # tfidf vectorizer of scikit learn
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000, max_df=0.5, use_idf=True,
                                         ngram_range=(1, 3))
        X = vectorizer.fit_transform(detokenized_doc)
        print(X.shape)  # check shape of the document-term matrixterms = vectorizer.get_feature_names()

        terms = vectorizer.get_feature_names()
        self.clusteringKmeans(X, terms)



    def clusteringKmeans(self, X, terms):
        num_clusters = 10
        km = KMeans(n_clusters=num_clusters)
        km.fit(X)

        U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=300,
                                      random_state=122)

        # printing the concepts
        for i, comp in enumerate(VT):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
            print("Concept " + str(i) + ": ")
            for t in sorted_terms:
                print(t[0])
            print(" ")


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