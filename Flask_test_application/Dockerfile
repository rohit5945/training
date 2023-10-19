FROM ubuntu:22.04
FROM python:3.8-slim-buster
RUN conda create -n myconda
RUN conda activate myconda
WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "app.py"]