FROM python:3.11.4-bookworm

RUN apt-get update && apt-get install libsasl2-dev libsasl2-modules libopencv-dev -y

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -U pip && \
    pip install -U setuptools && \
    pip install -r requirements.txt

