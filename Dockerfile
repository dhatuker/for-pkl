FROM python:3.7

WORKDIR /news-crawling-pkl

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY newsrunner.py .

COPY Lib/ .

COPY db/ .

COPY config/ .

COPY logs/ ./logs

EXPOSE 8080