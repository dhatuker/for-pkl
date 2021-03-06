import configparser
import logging

import records

class NewsparserDatabaseHandler(object):
    _instance = None
    _db = None
    _host = None
    _port = None
    _user = None
    _pass = None
    _dbname = None
    logger = None

    def getInstance(_host, _port, _user, _pass, _dbname):
        return NewsparserDatabaseHandler(_host, _port, _user, _pass, _dbname)

    def __init__(self, _host, _port, _user, _pass, _dbname):
        self._host = _host
        self._port = _port
        self._user = _user
        self._pass = _pass
        self._dbname = _dbname
        self.logger = logging.getLogger()
        self.connect()
        NewsparserDatabaseHandler._instance = self

    def setLogger(self, logger):
        self.logger = logger

    def connect(self):
        # try:
        self.logger.debug('connecting to MySQL database...')
        conn_string = 'mysql://{}:{}/{}?user={}&password={}&charset=utf8mb4'. \
            format(self._host, self._port, self._dbname, self._user, self._pass)
        self.logger.debug(conn_string)
        self._db = records.Database(conn_string)
        rs = self._db.query('SELECT VERSION() as ver', fetchall=True)
        if len(rs) > 0:
            db_version = rs[0].ver
        # except sqlalchemy.exc.OperationalError as error:
        #     self.logger.info('Error: connection not established {}'.format(error))
        NewsparserDatabaseHandler._instance = None
        # else:
        self.logger.debug('connection established: {}'.format(db_version))

    @staticmethod
    def instantiate_from_configparser(cfg, logger):
        if isinstance(cfg, configparser.ConfigParser):
            dbhandler = NewsparserDatabaseHandler.getInstance(cfg.get('Database', 'host'),
                                                               cfg.get('Database', 'port'),
                                                               cfg.get('Database', 'username'),
                                                               cfg.get('Database', 'password'),
                                                               cfg.get('Database', 'dbname'))
            dbhandler.setLogger(logger)
            return dbhandler
        else:
            raise Exception('cfg is not an instance of configparser')

    def insert_news(self, news_id, link, title, date, content):
        sql = """REPLACE INTO news_content (news_id, link, title, date, content)
        VALUES (:news_id, :link, :title, :date, :content)"""
        rs = self._db.query(sql, news_id=news_id, link=link, title=title, date=date, content=content)
        return rs

    def get_source(self, input):
        test = '%' + input + '%'
        sql = """SELECT * FROM news_source WHERE link LIKE :test"""
        rs = self._db.query(sql, test=test)
        return rs

    def get_article(self, input):
        time = input + '%'
        sql = """SELECT * FROM news_content WHERE date LIKE :time"""
        rs = self._db.query(sql, time=time)
        return rs

    def insert_prepro(self, date, news_num, news_word):
        sql = """REPLACE INTO news_prepro 
        (date, news_num, news_word)
        VALUES (:date, :news_num, :news_word)"""
        rs = self._db.query(sql, date=date, news_num=news_num, news_word=news_word)
        return rs

    def get_prepro(self, input):
        time = input + '%'
        sql = """SELECT * FROM news_prepro WHERE date LIKE :time"""
        rs = self._db.query(sql, time=time)
        return rs

    def insert_newstopic(self, document_no, dominant_topic, topic_perc, keywords, text, time):
        sql = """REPLACE INTO news_topic 
        (document_no, dominant_topic, topic_perc_contrib, keywords, text, date)
        VALUES (:document_no, :dominant_topic, :topic_perc, :keywords, :text, :time)"""
        rs = self._db.query(sql, document_no=document_no, dominant_topic=dominant_topic,
                            topic_perc=topic_perc, keywords=keywords, text=text, time=time)
        return rs

    def insert_newsdominant(self, dominant_topic, topic_keywords, num_doc, perc_doc, time):
        sql = """REPLACE INTO news_dominant_topic
        (dominant_topic, topic_keywords, num_doc, perc_doc, date)
        VALUES (:dominant_topic, :topic_keywords, :num_doc, :perc_doc, :time)"""
        rs = self._db.query(sql, dominant_topic=dominant_topic, topic_keywords=topic_keywords,
                            num_doc=num_doc, perc_doc=perc_doc, time=time)
        return rs

    def insert_newsrepre(self, topic_num, topik_perc, keyword, text, time):
        sql = """REPLACE INTO news_repre_doc
        (Topic_Num, Topic_Perc_Contrib, Keywords, Text, date)
        VALUES (:topic_num, :topik_perc, :keyword, :text, :time)"""
        rs = self._db.query(sql, topic_num=topic_num, topik_perc=topik_perc,
                            keyword=keyword, text=text, time=time)
        return rs