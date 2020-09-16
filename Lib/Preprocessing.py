import logbook
import time
import numpy as np
import re
import sys
import configparser
import os
import socket
from Lib.PreHelper import PreHelper
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import gensim
from gensim.models import Phrases
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array

import warnings

warnings.filterwarnings('ignore')
import pandas as pd

from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from db.NewsparserDatabaseHandler import NewsparserDatabaseHandler

class PreprocessingData(object):
    logger = None
    config = None
    time_stamp = None

    def __init__(self, db, config=None, logger=None, time_stamp=None):
        self.logger = logger
        self.config = config
        self.time_stamp = time_stamp
        self.db = db

        g = open("stopwords.txt", "r")
        self.gtext = g.read()
        g.close()

    def prepros(self):
        # today = str(date.today())
        today = '2020-09-08'
        data = list(self.db.get_article(today))
        stop_words = self.gtext.split("\n")
        stop_words.extend(['kompas','republika' ,'com' ,'co'])

        doc = list()

        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

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

        # Stemming
        news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: stemmer.stem(x))

        # tokenization
        tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

        # remove stop-words
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

        count_news = 0

        for news in tokenized_doc:
            count_news = count_news + 1
            for word in news:
                print(word)
                #self.db.insert_prepro(self.time_stamp, count_news, word)

        self.topicModeling(tokenized_doc)

    def topicModeling(self, text_list):
        bigram = Phrases(text_list, min_count=10)
        trigram = Phrases(bigram[text_list])
        for idx in range(len(text_list)):
            for token in bigram[text_list[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    text_list[idx].append(token)
            for token in trigram[text_list[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    text_list[idx].append(token)
        dictionary = corpora.Dictionary(text_list)
        dictionary.filter_extremes(no_below=5, no_above=0.2)
        # no_below (int, optional) – Keep tokens which are contained in at least no_below documents.
        # no_above (float, optional) – Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).
        print(dictionary)

        doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
        # The function doc2bow converts document (a list of words) into the bag-of-words format
        '''The function doc2bow() simply counts the number of occurrences of each distinct word, 
        converts the word to its integer word id and returns the result as a sparse vector. 
        The sparse vector [(0, 1), (1, 1)] therefore reads: in the document “Human computer interaction”, 
        the words computer (id 0) and human (id 1) appear once; 
        the other ten dictionary words appear (implicitly) zero times.'''
        print(len(doc_term_matrix))
        print(doc_term_matrix[100])
        tfidf = models.TfidfModel(doc_term_matrix)  # build TF-IDF model
        corpus_tfidf = tfidf[doc_term_matrix]

        start = 1
        limit = 21
        step = 1
        model_list, coherence_values = self.compute_coherence_values(dictionary, corpus=corpus_tfidf,
                                                                texts=text_list, start=start, limit=limit, step=step)

        optimal_model = model_list[3]
        model_topics = optimal_model.show_topics(formatted=False)
        print("model list : ", model_topics)
        # show graphs
        import matplotlib.pyplot as plt

        x = range(start, limit, step)
        
        # plt.plot(x, coherence_values)
        # plt.xlabel("Num Topics")
        # plt.ylabel("Coherence score")
        # plt.legend(("coherence_values"), loc='best')
        # plt.show()

        nilai = []

        for m, cv in zip(x, coherence_values):
            nilai.append([m, cv])
            print("Num Topics =", m, " has Coherence Value of", round(cv, 6))

        nilai.sort(reverse=True, key=PreHelper.maxCV)

        df_topic_sents_keywords = self.format_topics_sentences(ldamodel=optimal_model, corpus=corpus_tfidf, texts=dictionary)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        # Show
        df_dominant_topic.head(10)

        print("Sorted Highest:", nilai)
        print("Sorted Highest:", nilai[0][1])

        model = LdaModel(corpus=corpus_tfidf, id2word=dictionary,
                         num_topics=nilai[0][0])  # num topic menyesuaikan hasil dari coherence value paling tinggi

        for idx, topic in model.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx, topic))

        import pyLDAvis.gensim

        data = pyLDAvis.gensim.prepare(model, corpus_tfidf, dictionary)
        print(data)
        pyLDAvis.save_html(data, 'lda-gensim.html')


    def compute_coherence_values(self, dictionary, corpus, texts, limit, start, step):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=100)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def format_topics_sentences(self, ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)


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

        #init time_stamp
        self.time_stamp = time.gmtime()

        self.db = NewsparserDatabaseHandler.instantiate_from_configparser(self.config, self.logger)

    def run(self):
        start_time = time.time()
        self.init()
        self.hostname = socket.gethostname()
        self.hostip = socket.gethostbyname(self.hostname)
        self.logger.info("Starting {} on {}".format(type(self).__name__, self.hostname))
        self.PreprocessingData = PreprocessingData(db=self.db, config=self.config, logger=self.logger, time_stamp=self.time_stamp)

        self.PreprocessingData.prepros()

        self.logger.info("Finish %s" % self.filename)
        print("--- %s seconds ---" % (time.time() - start_time))