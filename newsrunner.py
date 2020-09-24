import sys

from NewsParser import NewsParserData, NewsParsing

def main():
    news = NewsParsing()
    news.run(sys.argv[1])
    del news

if __name__ == '__main__':
    main()
