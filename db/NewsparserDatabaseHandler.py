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

    def insert_news(self, title, content, tgl_terbit, editor, link):
        sql = """REPLACE INTO content_table (title, content, tgl_terbit, editor, link)
        VALUES (:title, :content, :tgl_terbit, :editor, :link)"""
        rs = self._db.query(sql, title=title, content=content, tgl_terbit=tgl_terbit, editor=editor, link=link)
        return rs
